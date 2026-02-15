#!/usr/bin/env python3
"""Document Control Email Monitor - Web Dashboard.

Flask web app with APScheduler background scanning.
Phase 1 uses scan_and_queue() for the review queue workflow.
Phase 2 is user-driven approve/reject from the dashboard.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime

from flask import Flask, Response, jsonify, render_template, request

import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)


def _summarize_body(body, max_chars=1500):
    """Clean raw email body into a readable summary.

    Strips forwarded headers, signatures, URLs, and excessive whitespace.
    """
    if not body:
        return ''

    lines = body.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    cleaned = []
    skip_rest = False

    for line in lines:
        stripped = line.strip()

        # Stop at common signature / forward markers
        if stripped in ('--', '---', '____', '___') or \
           stripped.startswith('________________________________') or \
           stripped.startswith('Sent from Outlook') or \
           stripped.startswith('Sent from my ') or \
           stripped.startswith('Get Outlook for'):
            skip_rest = True
            continue

        if skip_rest:
            # After a forward marker, look for the actual forwarded content
            # Stop skipping if we hit a "From:" line (start of forwarded message)
            if re.match(r'^From:\s', stripped):
                skip_rest = False
                cleaned.append('\n--- Forwarded ---')
                cleaned.append(stripped)
            continue

        # Skip lines that are just URLs
        if re.match(r'^https?://\S+$', stripped):
            continue

        # Strip inline URLs but keep surrounding text
        line_clean = re.sub(r'<https?://[^>]+>', '', stripped)
        line_clean = line_clean.strip()

        if line_clean:
            cleaned.append(line_clean)
        elif cleaned and cleaned[-1] != '':
            cleaned.append('')  # Preserve one blank line between paragraphs

    # Join and collapse multiple blank lines
    result = '\n'.join(cleaned).strip()
    result = re.sub(r'\n{3,}', '\n\n', result)

    # Truncate if still too long
    if len(result) > max_chars:
        result = result[:max_chars].rsplit('\n', 1)[0] + '\n\n[... truncated]'

    return result

app = Flask(__name__)

# --- Scan state (shared across threads) ---
scan_lock = threading.Lock()
scan_state = {
    'status': 'idle',        # idle | scanning | error
    'last_scan_time': None,
    'last_scan_result': None, # e.g. "3 emails queued" or error message
    'last_scan_ok': True,
    'next_scan_time': None,
}

# --- Logging setup (run once) ---
_logging_initialized = False


def _ensure_logging():
    """Set up logging once, then no-op."""
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    from email_interface.config import load_config, resolve_path
    email_cfg = load_config(resolve_path('config/email_config.yaml', BASE_DIR))
    log_config = email_cfg.get('logging', {})

    # Use email_monitor's setup_logging for the first call
    import email_monitor
    email_monitor.setup_logging(log_config)

    # Monkey-patch so subsequent calls don't add duplicate handlers
    email_monitor.setup_logging = lambda *args, **kwargs: None


def _get_db_path():
    return os.path.join(BASE_DIR, 'data', 'tracking.db')


def _get_log_path():
    return os.path.join(BASE_DIR, 'logs', 'email_interface.log')


def _get_interval_minutes():
    from email_interface.config import load_config, resolve_path
    email_cfg = load_config(resolve_path('config/email_config.yaml', BASE_DIR))
    return email_cfg.get('polling', {}).get('interval_minutes', 10)


def _get_tracker():
    """Create a ProcessingTracker instance."""
    from email_interface.persistence import ProcessingTracker
    from email_interface.config import load_config, resolve_path
    class_cfg = load_config(resolve_path('config/classification_config.yaml', BASE_DIR))
    prefix = class_cfg.get('transmittal', {}).get('prefix', 'TRN')
    return ProcessingTracker(db_path=_get_db_path(), prefix=prefix)


def _get_class_cfg():
    """Load classification config."""
    from email_interface.config import load_config, resolve_path
    return load_config(resolve_path('config/classification_config.yaml', BASE_DIR))


def _count_messages(db_path):
    """Count total processed messages in SQLite."""
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM processed_messages")
        return cur.fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def run_scan():
    """Execute a scan via scan_and_queue(), tracking state."""
    if not scan_lock.acquire(blocking=False):
        return  # Already running

    try:
        scan_state['status'] = 'scanning'
        _ensure_logging()
        logger = logging.getLogger(__name__)

        logger.info("Dashboard: starting scan (scan_and_queue)")
        from email_interface.scanner import scan_and_queue
        counts = scan_and_queue(base_dir=BASE_DIR)

        scan_state['status'] = 'idle'
        scan_state['last_scan_time'] = datetime.now().isoformat()
        scan_state['last_scan_result'] = (
            f"{counts['queued']} queued, {counts['out_of_scope']} out-of-scope, "
            f"{counts['skipped']} skipped"
        )
        scan_state['last_scan_ok'] = counts['errors'] == 0
        logger.info("Dashboard: scan complete - %s", scan_state['last_scan_result'])

    except Exception as e:
        scan_state['status'] = 'error'
        scan_state['last_scan_time'] = datetime.now().isoformat()
        scan_state['last_scan_result'] = str(e)
        scan_state['last_scan_ok'] = False
        logging.getLogger(__name__).error("Dashboard: scan failed - %s", e, exc_info=True)
    finally:
        scan_lock.release()


def run_scan_in_thread():
    """Launch scan in a background thread."""
    t = threading.Thread(target=run_scan, daemon=True)
    t.start()


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    return jsonify({
        'status': scan_state['status'],
        'last_scan_time': scan_state['last_scan_time'],
        'last_scan_result': scan_state['last_scan_result'],
        'last_scan_ok': scan_state['last_scan_ok'],
        'next_scan_time': scan_state['next_scan_time'],
    })


ALLOWED_CLASSIFIER_METHODS = ['auto', 'rule_based', 'gemini', 'groq', 'claude_api']


def _resolve_classifier_method():
    """Return (method, source) — reads SQLite override first, falls back to YAML."""
    tracker = _get_tracker()
    try:
        override = tracker.get_setting('classifier_method')
    finally:
        tracker.close()
    if override and override in ALLOWED_CLASSIFIER_METHODS:
        return override, 'dashboard'
    class_cfg = _get_class_cfg()
    return class_cfg.get('classifier', {}).get('method', 'rule_based'), 'config'


def _method_active_label(method, keys):
    """Build a human-readable label for the active classifier method."""
    if method == 'auto':
        active = []
        if keys['gemini']:
            active.append('Gemini')
        if keys['groq']:
            active.append('Groq')
        if keys['claude']:
            active.append('Claude')
        active.append('Keywords')
        if len(active) == 1:
            return 'Keywords (no AI keys detected)'
        return ' > '.join(active)
    elif method == 'rule_based':
        return 'Keywords'
    elif method == 'claude_api':
        return 'Claude API' if keys['claude'] else 'Claude API (key missing!)'
    elif method == 'gemini':
        return 'Gemini' if keys['gemini'] else 'Gemini (key missing!)'
    elif method == 'groq':
        return 'Groq' if keys['groq'] else 'Groq (key missing!)'
    return method


@app.route('/api/classifier-info')
def api_classifier_info():
    """Return which classifier method is active and which AI keys are detected."""
    method, source = _resolve_classifier_method()
    keys = {
        'gemini': bool(os.environ.get('GEMINI_API_KEY')),
        'groq': bool(os.environ.get('GROQ_API_KEY')),
        'claude': bool(os.environ.get('ANTHROPIC_API_KEY')),
    }
    yaml_default = _get_class_cfg().get('classifier', {}).get('method', 'rule_based')

    return jsonify({
        'method': method,
        'source': source,
        'yaml_default': yaml_default,
        'active_label': _method_active_label(method, keys),
        'keys': keys,
    })


@app.route('/api/settings/classifier-method', methods=['GET'])
def api_get_classifier_method():
    """Return current classifier method setting."""
    method, source = _resolve_classifier_method()
    yaml_default = _get_class_cfg().get('classifier', {}).get('method', 'rule_based')
    return jsonify({'method': method, 'source': source, 'yaml_default': yaml_default})


@app.route('/api/settings/classifier-method', methods=['PUT'])
def api_set_classifier_method():
    """Set classifier method override in SQLite."""
    data = request.get_json(silent=True) or {}
    method = data.get('method', '').strip()
    if method not in ALLOWED_CLASSIFIER_METHODS:
        return jsonify({'error': f'Invalid method. Allowed: {ALLOWED_CLASSIFIER_METHODS}'}), 400
    tracker = _get_tracker()
    try:
        tracker.set_setting('classifier_method', method)
    finally:
        tracker.close()
    keys = {
        'gemini': bool(os.environ.get('GEMINI_API_KEY')),
        'groq': bool(os.environ.get('GROQ_API_KEY')),
        'claude': bool(os.environ.get('ANTHROPIC_API_KEY')),
    }
    return jsonify({
        'message': f'Classifier method set to {method}',
        'method': method,
        'source': 'dashboard',
        'active_label': _method_active_label(method, keys),
    })


@app.route('/api/settings/ai-instructions', methods=['GET'])
def api_get_ai_instructions():
    """Return current custom AI instructions."""
    tracker = _get_tracker()
    try:
        instructions = tracker.get_setting('classifier_instructions') or ''
        # Get updated_at from settings table
        conn = sqlite3.connect(_get_db_path())
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT updated_at FROM settings WHERE key = 'classifier_instructions'")
        row = cur.fetchone()
        conn.close()
        updated_at = row['updated_at'] if row else None
        return jsonify({'instructions': instructions, 'updated_at': updated_at})
    finally:
        tracker.close()


@app.route('/api/settings/ai-instructions', methods=['PUT'])
def api_set_ai_instructions():
    """Save custom AI instructions."""
    data = request.get_json(silent=True) or {}
    instructions = data.get('instructions', '').strip()
    tracker = _get_tracker()
    try:
        tracker.set_setting('classifier_instructions', instructions)
        return jsonify({'message': 'AI instructions saved', 'instructions': instructions})
    finally:
        tracker.close()


@app.route('/api/home-stats')
def api_home_stats():
    """Return summary stats for the home page."""
    tracker = _get_tracker()
    try:
        stats = tracker.get_pending_stats()
        departments = tracker.get_departments(active_only=True)
        contacts = tracker.get_contacts(active_only=True)
    finally:
        tracker.close()
    total_processed = _count_messages(_get_db_path())
    return jsonify({
        'pending_review': stats.get('pending_review', 0),
        'total_processed': total_processed,
        'departments': len(departments),
        'contacts': len(contacts),
    })


@app.route('/api/scan', methods=['POST'])
def api_scan():
    if scan_state['status'] == 'scanning':
        return jsonify({'message': 'Scan already in progress'}), 409

    run_scan_in_thread()
    return jsonify({'message': 'Scan started'}), 202


@app.route('/api/emails')
def api_emails():
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return jsonify({'emails': [], 'total': 0, 'today': 0, 'with_attachments': 0})

    tracker = _get_tracker()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Recent emails with full classification (newest first, limit 50)
        cur = conn.execute(
            """SELECT transmittal_no, processed_at, sender, sender_name,
                      to_recipients, cc_recipients, subject, attachment_count,
                      doc_type, discipline, department, response_required, references_json
               FROM processed_messages ORDER BY processed_at DESC LIMIT 50"""
        )
        emails = []
        for row in cur.fetchall():
            e = dict(row)
            # Look up sender in contacts for company info
            contact = tracker.find_contact_by_email(e.get('sender', ''))
            e['sender_company'] = contact['company'] if contact else ''
            e['sender_known'] = contact is not None
            emails.append(e)

        # Stats
        total = conn.execute("SELECT COUNT(*) FROM processed_messages").fetchone()[0]

        today_str = datetime.now().strftime('%Y-%m-%d')
        today = conn.execute(
            "SELECT COUNT(*) FROM processed_messages WHERE processed_at LIKE ?",
            (today_str + '%',)
        ).fetchone()[0]

        with_att = conn.execute(
            "SELECT COUNT(*) FROM processed_messages WHERE attachment_count > 0"
        ).fetchone()[0]

        return jsonify({
            'emails': emails,
            'total': total,
            'today': today,
            'with_attachments': with_att,
        })
    except sqlite3.OperationalError:
        return jsonify({'emails': [], 'total': 0, 'today': 0, 'with_attachments': 0})
    finally:
        conn.close()
        tracker.close()


@app.route('/api/log')
def api_log():
    log_path = _get_log_path()
    lines = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                # Read last 50 lines efficiently
                all_lines = deque(f, maxlen=50)
                lines = list(all_lines)
        except OSError:
            lines = ['[Could not read log file]']
    return jsonify({'lines': lines})


# --- Pending / Review Queue Routes ---

@app.route('/api/pending')
def api_pending():
    status = request.args.get('status', 'pending_review')
    tracker = _get_tracker()
    try:
        pending = tracker.get_pending_emails(status=status)
        stats = tracker.get_pending_stats()
        # Enrich with contact info
        for email in pending:
            contact = tracker.find_contact_by_email(email.get('sender', ''))
            email['sender_company'] = contact['company'] if contact else ''
            email['sender_known'] = contact is not None
        return jsonify({'emails': pending, 'stats': stats})
    finally:
        tracker.close()


@app.route('/api/pending/<int:pending_id>/detail')
def api_pending_detail(pending_id):
    """Return full email record including body for the detail modal."""
    tracker = _get_tracker()
    try:
        # Use get_pending_email (singular) which does SELECT * and includes body
        email = tracker.get_pending_email(pending_id)
        if not email:
            return jsonify({'error': 'Not found'}), 404

        # Enrich with contact info
        contact = tracker.find_contact_by_email(email.get('sender', ''))
        email['sender_company'] = contact['company'] if contact else ''
        email['sender_known'] = contact is not None

        # Get attachment metadata with signature detection
        attachments = tracker.get_pending_attachments_meta(pending_id)
        from email_interface.attachment_handler import AttachmentHandler
        from email_interface.config import load_config, resolve_path
        email_cfg = load_config(resolve_path('config/email_config.yaml', BASE_DIR))
        att_cfg = email_cfg.get('attachments', {})
        handler = AttachmentHandler(
            base_path=resolve_path(att_cfg.get('base_path', 'Attachments'), BASE_DIR),
            max_size_mb=att_cfg.get('max_size_mb', 25),
            allowed_extensions=att_cfg.get('allowed_extensions'),
            skip_signature_attachments=att_cfg.get('skip_signature_attachments', True),
        )
        for att in attachments:
            att['is_signature'] = handler.is_signature_attachment(
                att.get('filename', ''), att.get('content_type', ''), att.get('size')
            )
        email['attachments'] = attachments

        # Add cleaned body summary
        email['body_summary'] = _summarize_body(email.get('body', ''))

        return jsonify(email)
    finally:
        tracker.close()


@app.route('/api/pending/<int:pending_id>/approve', methods=['POST'])
def api_approve(pending_id):
    edits = request.get_json(silent=True) or {}
    tracker = _get_tracker()
    try:
        result = tracker.approve_pending_email(pending_id, edits=edits)
        transmittal_no = result['transmittal_no']
        pe = result['pending_email']

        # Save attachments from blobs (with signature filtering)
        attachments = tracker.get_pending_attachments(pending_id)
        saved_files = []
        if attachments:
            from email_interface.attachment_handler import AttachmentHandler
            from email_interface.config import load_config, resolve_path
            email_cfg = load_config(resolve_path('config/email_config.yaml', BASE_DIR))
            att_cfg = email_cfg.get('attachments', {})
            handler = AttachmentHandler(
                base_path=resolve_path(att_cfg.get('base_path', 'Attachments'), BASE_DIR),
                max_size_mb=att_cfg.get('max_size_mb', 25),
                allowed_extensions=att_cfg.get('allowed_extensions'),
                skip_signature_attachments=att_cfg.get('skip_signature_attachments', True),
            )
            for att in attachments:
                # Skip signature images
                if handler.is_signature_attachment(
                    att['filename'], att['content_type'], att.get('size') or len(att.get('data', b''))
                ):
                    logger.info("Skipping signature attachment: %s", att['filename'])
                    continue
                result_att = handler.save_single_attachment(
                    filename=att['filename'],
                    data=att['data'],
                    content_type=att['content_type'],
                    email_date=pe.get('email_date'),
                    transmittal_no=transmittal_no,
                )
                if result_att:
                    saved_files.append(result_att)

        # Update attachment_folder on the pending record
        att_folder = ''
        if saved_files:
            att_folder = os.path.dirname(saved_files[0]['path'])
            conn = sqlite3.connect(_get_db_path())
            conn.execute(
                "UPDATE pending_emails SET attachment_folder = ? WHERE id = ?",
                (att_folder, pending_id),
            )
            conn.commit()
            conn.close()

        # Log to Excel
        from email_monitor import update_excel_log
        from email_interface.config import load_config, resolve_path
        email_cfg = load_config(resolve_path('config/email_config.yaml', BASE_DIR))
        excel_path = resolve_path(
            email_cfg.get('excel', {}).get('output_path', 'logs/correspondence_log.xlsx'),
            BASE_DIR,
        )

        to_list = json.loads(pe.get('to_recipients', '[]'))
        cc_list = json.loads(pe.get('cc_recipients', '[]'))
        refs = json.loads(pe.get('references_json', '[]'))

        excel_entry = {
            'transmittal_no': transmittal_no,
            'date': pe.get('email_date'),
            'sender': pe.get('sender', ''),
            'to': to_list,
            'cc': cc_list,
            'subject': pe.get('subject', ''),
            'references': refs,
            'doc_type': pe.get('doc_type', ''),
            'discipline': pe.get('discipline', ''),
            'response_required': pe.get('response_required'),
            'attachments': [{}] * pe.get('attachment_count', 0),
            'attachments_saved': saved_files,
            'message_id': pe.get('message_id', ''),
        }
        update_excel_log([excel_entry], excel_path)

        return jsonify({
            'message': f'Approved as {transmittal_no}',
            'transmittal_no': transmittal_no,
            'attachment_folder': att_folder,
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    finally:
        tracker.close()


@app.route('/api/pending/<int:pending_id>/reject', methods=['POST'])
def api_reject(pending_id):
    tracker = _get_tracker()
    try:
        tracker.reject_pending_email(pending_id)
        return jsonify({'message': 'Rejected'})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    finally:
        tracker.close()


@app.route('/api/pending/<int:pending_id>/move-to-review', methods=['POST'])
def api_move_to_review(pending_id):
    tracker = _get_tracker()
    try:
        tracker.move_to_review(pending_id)
        return jsonify({'message': 'Moved to review'})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    finally:
        tracker.close()


@app.route('/api/pending/<int:pending_id>/attachments')
def api_pending_attachments(pending_id):
    """Return attachment metadata (filenames, sizes) with signature detection."""
    tracker = _get_tracker()
    try:
        attachments = tracker.get_pending_attachments_meta(pending_id)
        # Add signature detection
        from email_interface.attachment_handler import AttachmentHandler
        from email_interface.config import load_config, resolve_path
        email_cfg = load_config(resolve_path('config/email_config.yaml', BASE_DIR))
        att_cfg = email_cfg.get('attachments', {})
        handler = AttachmentHandler(
            base_path=resolve_path(att_cfg.get('base_path', 'Attachments'), BASE_DIR),
            max_size_mb=att_cfg.get('max_size_mb', 25),
            allowed_extensions=att_cfg.get('allowed_extensions'),
            skip_signature_attachments=att_cfg.get('skip_signature_attachments', True),
        )
        for att in attachments:
            att['is_signature'] = handler.is_signature_attachment(
                att.get('filename', ''), att.get('content_type', ''), att.get('size')
            )
        return jsonify({'attachments': attachments})
    finally:
        tracker.close()


@app.route('/api/pending/<int:pending_id>/attachments/<int:att_id>')
def api_pending_attachment_download(pending_id, att_id):
    """Serve a single pending attachment as a download."""
    tracker = _get_tracker()
    try:
        att = tracker.get_single_pending_attachment(att_id, pending_id)
        if not att:
            return jsonify({'error': 'Attachment not found'}), 404
        return Response(
            att['data'],
            mimetype=att['content_type'] or 'application/octet-stream',
            headers={
                'Content-Disposition': f'inline; filename="{att["filename"]}"',
                'Content-Length': str(att['size'] or len(att['data'])),
            },
        )
    finally:
        tracker.close()


@app.route('/api/pending/bulk-approve', methods=['POST'])
def api_bulk_approve():
    data = request.get_json(silent=True) or {}
    ids = data.get('ids', [])
    if not ids:
        return jsonify({'error': 'No ids provided'}), 400

    results = []
    errors = []
    tracker = _get_tracker()
    try:
        for pid in ids:
            try:
                result = tracker.approve_pending_email(pid)
                transmittal_no = result['transmittal_no']
                pe = result['pending_email']

                # Save attachments (with signature filtering)
                attachments = tracker.get_pending_attachments(pid)
                saved_files = []
                if attachments:
                    from email_interface.attachment_handler import AttachmentHandler
                    from email_interface.config import load_config, resolve_path
                    email_cfg = load_config(resolve_path('config/email_config.yaml', BASE_DIR))
                    att_cfg = email_cfg.get('attachments', {})
                    handler = AttachmentHandler(
                        base_path=resolve_path(att_cfg.get('base_path', 'Attachments'), BASE_DIR),
                        max_size_mb=att_cfg.get('max_size_mb', 25),
                        allowed_extensions=att_cfg.get('allowed_extensions'),
                        skip_signature_attachments=att_cfg.get('skip_signature_attachments', True),
                    )
                    for att in attachments:
                        if handler.is_signature_attachment(
                            att['filename'], att['content_type'],
                            att.get('size') or len(att.get('data', b''))
                        ):
                            continue
                        result_att = handler.save_single_attachment(
                            filename=att['filename'],
                            data=att['data'],
                            content_type=att['content_type'],
                            email_date=pe.get('email_date'),
                            transmittal_no=transmittal_no,
                        )
                        if result_att:
                            saved_files.append(result_att)

                att_folder = ''
                if saved_files:
                    att_folder = os.path.dirname(saved_files[0]['path'])

                # Log to Excel
                from email_monitor import update_excel_log
                from email_interface.config import load_config, resolve_path
                email_cfg = load_config(resolve_path('config/email_config.yaml', BASE_DIR))
                excel_path = resolve_path(
                    email_cfg.get('excel', {}).get('output_path', 'logs/correspondence_log.xlsx'),
                    BASE_DIR,
                )
                to_list = json.loads(pe.get('to_recipients', '[]'))
                cc_list = json.loads(pe.get('cc_recipients', '[]'))
                refs = json.loads(pe.get('references_json', '[]'))

                excel_entry = {
                    'transmittal_no': transmittal_no,
                    'date': pe.get('email_date'),
                    'sender': pe.get('sender', ''),
                    'to': to_list,
                    'cc': cc_list,
                    'subject': pe.get('subject', ''),
                    'references': refs,
                    'doc_type': pe.get('doc_type', ''),
                    'discipline': pe.get('discipline', ''),
                    'response_required': pe.get('response_required'),
                    'attachments': [{}] * pe.get('attachment_count', 0),
                    'attachments_saved': saved_files,
                    'message_id': pe.get('message_id', ''),
                }
                update_excel_log([excel_entry], excel_path)

                results.append({'id': pid, 'transmittal_no': transmittal_no})
            except Exception as e:
                errors.append({'id': pid, 'error': str(e)})
    finally:
        tracker.close()

    return jsonify({'approved': results, 'errors': errors})


@app.route('/api/pending/bulk-reject', methods=['POST'])
def api_bulk_reject():
    data = request.get_json(silent=True) or {}
    ids = data.get('ids', [])
    if not ids:
        return jsonify({'error': 'No ids provided'}), 400

    rejected = []
    errors = []
    tracker = _get_tracker()
    try:
        for pid in ids:
            try:
                tracker.reject_pending_email(pid)
                rejected.append(pid)
            except Exception as e:
                errors.append({'id': pid, 'error': str(e)})
    finally:
        tracker.close()

    return jsonify({'rejected': rejected, 'errors': errors})


# --- Department Routes ---

@app.route('/api/departments', methods=['GET'])
def api_departments_list():
    tracker = _get_tracker()
    try:
        departments = tracker.get_departments()
        return jsonify({'departments': departments})
    finally:
        tracker.close()


@app.route('/api/departments', methods=['POST'])
def api_departments_add():
    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    keywords = data.get('keywords', [])
    description = data.get('description', '').strip()
    if not name:
        return jsonify({'error': 'Name is required'}), 400

    tracker = _get_tracker()
    try:
        dept_id = tracker.add_department(name, keywords, description=description)
        return jsonify({'id': dept_id, 'message': f'Department "{name}" added'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': f'Department "{name}" already exists'}), 409
    finally:
        tracker.close()


@app.route('/api/departments/<int:dept_id>', methods=['PUT'])
def api_departments_update(dept_id):
    data = request.get_json(silent=True) or {}
    tracker = _get_tracker()
    try:
        tracker.update_department(
            dept_id,
            name=data.get('name'),
            keywords=data.get('keywords'),
            is_active=data.get('is_active'),
            description=data.get('description'),
        )
        return jsonify({'message': 'Updated'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Department name already exists'}), 409
    finally:
        tracker.close()


@app.route('/api/departments/<int:dept_id>', methods=['DELETE'])
def api_departments_delete(dept_id):
    tracker = _get_tracker()
    try:
        tracker.delete_department(dept_id)
        return jsonify({'message': 'Deleted'})
    finally:
        tracker.close()


# --- Contact Routes (expected senders) ---

@app.route('/api/contacts', methods=['GET'])
def api_contacts_list():
    tracker = _get_tracker()
    try:
        contacts = tracker.get_contacts()
        return jsonify({'contacts': contacts})
    finally:
        tracker.close()


@app.route('/api/contacts', methods=['POST'])
def api_contacts_add():
    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    if not name or not email:
        return jsonify({'error': 'Name and email are required'}), 400

    tracker = _get_tracker()
    try:
        contact_id = tracker.add_contact(
            name=name,
            email=email,
            company=data.get('company', '').strip(),
            department=data.get('department', '').strip(),
            notes=data.get('notes', '').strip(),
        )
        return jsonify({'id': contact_id, 'message': f'Contact "{name}" added'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': f'Contact with email "{email}" already exists'}), 409
    finally:
        tracker.close()


@app.route('/api/contacts/<int:contact_id>', methods=['PUT'])
def api_contacts_update(contact_id):
    data = request.get_json(silent=True) or {}
    tracker = _get_tracker()
    try:
        tracker.update_contact(
            contact_id,
            name=data.get('name'),
            email=data.get('email'),
            company=data.get('company'),
            department=data.get('department'),
            notes=data.get('notes'),
            is_active=data.get('is_active'),
        )
        return jsonify({'message': 'Updated'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists for another contact'}), 409
    finally:
        tracker.close()


@app.route('/api/contacts/<int:contact_id>', methods=['DELETE'])
def api_contacts_delete(contact_id):
    tracker = _get_tracker()
    try:
        tracker.delete_contact(contact_id)
        return jsonify({'message': 'Deleted'})
    finally:
        tracker.close()


# --- Classification Options (for dropdowns) ---

@app.route('/api/classification-options')
def api_classification_options():
    class_cfg = _get_class_cfg()
    tracker = _get_tracker()
    try:
        departments = tracker.get_departments(active_only=True)
        return jsonify({
            'doc_types': list(class_cfg.get('type_keywords', {}).keys()),
            'disciplines': list(class_cfg.get('discipline_keywords', {}).keys()),
            'departments': [d['name'] for d in departments],
        })
    finally:
        tracker.close()


# --- Scheduler ---

def start_scheduler():
    """Start APScheduler to run scans at the configured interval."""
    from apscheduler.schedulers.background import BackgroundScheduler

    interval = _get_interval_minutes()

    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        run_scan,
        'interval',
        minutes=interval,
        id='email_scan',
        next_run_time=None,  # Don't run immediately on startup
    )
    scheduler.start()

    # Track next scan time
    def update_next_scan():
        job = scheduler.get_job('email_scan')
        if job and job.next_run_time:
            scan_state['next_scan_time'] = job.next_run_time.isoformat()

    # Update next_scan_time after each job run
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
    def on_job_done(event):
        update_next_scan()
    scheduler.add_listener(on_job_done, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    # Reschedule so first run happens after `interval` minutes from now
    scheduler.reschedule_job('email_scan', trigger='interval', minutes=interval)
    update_next_scan()

    logging.getLogger(__name__).info(
        "Scheduler started: scanning every %d minutes", interval
    )


# --- Main ---

if __name__ == '__main__':
    _ensure_logging()

    # Ensure departments are seeded on first run
    tracker = _get_tracker()
    class_cfg = _get_class_cfg()
    tracker.seed_default_departments(class_cfg.get('discipline_keywords', {}))
    tracker.close()

    start_scheduler()

    print(f"\n  Document Control Dashboard running at http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
