#!/usr/bin/env python3
"""Document Control Email Monitor - Web Dashboard.

Flask web app with APScheduler background scanning.
Phase 1 uses scan_and_queue() for the review queue workflow.
Phase 2 is user-driven approve/reject from the dashboard.
"""

import io
import json
import logging
import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timedelta

from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from openpyxl import Workbook

import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

# Only show document attachments in the UI (filter out images, signatures, etc.)
DOCUMENT_EXTENSIONS = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                       '.dwg', '.dxf', '.msg', '.zip', '.rar', '.7z', '.csv', '.txt'}


def _is_document(filename):
    """Check if a filename has a document extension."""
    if not filename:
        return False
    ext = os.path.splitext(filename.lower())[1]
    return ext in DOCUMENT_EXTENSIONS


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

# --- Auth (F1) ---
DASHBOARD_PASSWORD = os.environ.get('DASHBOARD_PASSWORD', '')
_secret = os.environ.get('SECRET_KEY')
if not _secret:
    logger.warning("SECRET_KEY not set — sessions will reset on app restart. Set SECRET_KEY env var for persistence.")
    _secret = os.urandom(32)
app.secret_key = _secret

if not DASHBOARD_PASSWORD:
    logger.warning("DASHBOARD_PASSWORD not set — dashboard is accessible without authentication.")


@app.before_request
def require_login():
    """If DASHBOARD_PASSWORD is set, require authentication."""
    if not DASHBOARD_PASSWORD:
        return  # No password set, skip auth
    if request.endpoint in ('login', 'static'):
        return  # Allow login page and static files
    if not session.get('authenticated'):
        return redirect(url_for('login'))


# --- Scan state (shared across threads) ---
scan_lock = threading.Lock()
scan_state = {
    'status': 'idle',        # idle | scanning | error
    'last_scan_time': None,
    'last_scan_result': None, # e.g. "3 emails queued" or error message
    'last_scan_ok': True,
    'next_scan_time': None,
    'classifiers_used': [],   # F2: track which classifiers were used in last scan
}

# --- Scan progress (real-time progress for UI) ---
scan_progress = {
    'active': False, 'total': 0, 'scanned': 0,
    'queued': 0, 'skipped': 0, 'out_of_scope': 0,
    'errors': 0, 'auto_assigned': 0, 'inherited': 0,
    'already_known': 0, 'auto_approved': 0,
    'current_subject': '', 'phase': '',
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


def _ensure_backfill():
    """Run one-time conversation backfill if not yet done."""
    try:
        tracker = _get_tracker()
        if not tracker.get_setting('conversations_backfilled'):
            logger.info("Running one-time conversation backfill...")
            tracker.backfill_conversations()
            tracker.set_setting('conversations_backfilled', '1')
            logger.info("Conversation backfill complete.")
        tracker.close()
    except Exception as e:
        logger.error("Backfill error: %s", e)


def _reset_scan_progress():
    """Reset scan progress dict for a new scan."""
    scan_progress['active'] = True
    scan_progress['total'] = 0
    scan_progress['scanned'] = 0
    scan_progress['queued'] = 0
    scan_progress['skipped'] = 0
    scan_progress['out_of_scope'] = 0
    scan_progress['errors'] = 0
    scan_progress['auto_assigned'] = 0
    scan_progress['inherited'] = 0
    scan_progress['already_known'] = 0
    scan_progress['auto_approved'] = 0
    scan_progress['current_subject'] = ''
    scan_progress['phase'] = ''


def _scan_progress_callback(counts, subject, phase=''):
    """Callback from scanner to update real-time progress."""
    scan_progress['scanned'] = counts.get('scanned', 0)
    scan_progress['total'] = counts.get('total', 0)
    scan_progress['queued'] = counts.get('queued', 0)
    scan_progress['skipped'] = counts.get('skipped', 0)
    scan_progress['out_of_scope'] = counts.get('out_of_scope', 0)
    scan_progress['errors'] = counts.get('errors', 0)
    scan_progress['auto_assigned'] = counts.get('auto_assigned', 0)
    scan_progress['inherited'] = counts.get('inherited', 0)
    scan_progress['already_known'] = counts.get('already_known', 0)
    scan_progress['auto_approved'] = counts.get('auto_approved', 0)
    scan_progress['current_subject'] = subject or ''
    if phase:
        scan_progress['phase'] = phase


def run_scan():
    """Execute a scan via scan_and_queue(), tracking state."""
    if not scan_lock.acquire(blocking=False):
        return  # Already running

    try:
        scan_state['status'] = 'scanning'
        _reset_scan_progress()
        _ensure_logging()
        logger = logging.getLogger(__name__)

        # Ensure conversation backfill on first scan
        _ensure_backfill()

        logger.info("Dashboard: starting scan (scan_and_queue)")
        from email_interface.scanner import scan_and_queue
        counts = scan_and_queue(base_dir=BASE_DIR, progress_callback=_scan_progress_callback)

        # F2: Check classifier methods used by recently scanned emails
        classifiers_used = []
        if counts['queued'] > 0 or counts['out_of_scope'] > 0:
            try:
                tracker = _get_tracker()
                conn = sqlite3.connect(_get_db_path())
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    """SELECT DISTINCT classifier_method FROM pending_emails
                       WHERE scanned_at >= ? AND classifier_method IS NOT NULL AND classifier_method != ''""",
                    (scan_state.get('last_scan_time') or '2000-01-01',)
                )
                classifiers_used = [row['classifier_method'] for row in cur.fetchall()]
                conn.close()
                tracker.close()
            except Exception:
                pass

        scan_state['status'] = 'idle'
        scan_state['last_scan_time'] = datetime.now().isoformat()
        auto_str = f", {counts['auto_assigned']} auto-assigned" if counts.get('auto_assigned') else ''
        inherited_str = f", {counts['inherited']} inherited" if counts.get('inherited') else ''
        auto_appr_str = f", {counts['auto_approved']} auto-approved" if counts.get('auto_approved') else ''
        scan_state['last_scan_result'] = (
            f"{counts['queued']} queued{auto_appr_str}{auto_str}{inherited_str}, "
            f"{counts['out_of_scope']} out-of-scope, {counts['skipped']} skipped"
        )
        scan_state['last_scan_ok'] = counts['errors'] == 0
        scan_state['classifiers_used'] = classifiers_used
        scan_progress['active'] = False
        logger.info("Dashboard: scan complete - %s", scan_state['last_scan_result'])

    except Exception as e:
        scan_state['status'] = 'error'
        scan_state['last_scan_time'] = datetime.now().isoformat()
        scan_state['last_scan_result'] = str(e)
        scan_state['last_scan_ok'] = False
        scan_progress['active'] = False
        logging.getLogger(__name__).error("Dashboard: scan failed - %s", e, exc_info=True)
    finally:
        scan_lock.release()


def run_scan_in_thread():
    """Launch scan in a background thread."""
    t = threading.Thread(target=run_scan, daemon=True)
    t.start()


# --- Flask Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if not DASHBOARD_PASSWORD:
        return redirect(url_for('index'))
    error = ''
    if request.method == 'POST':
        if request.form.get('password') == DASHBOARD_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('index'))
        error = 'Incorrect password'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
def index():
    return render_template('dashboard.html', auth_enabled=bool(DASHBOARD_PASSWORD))


@app.route('/api/status')
def api_status():
    return jsonify({
        'status': scan_state['status'],
        'last_scan_time': scan_state['last_scan_time'],
        'last_scan_result': scan_state['last_scan_result'],
        'last_scan_ok': scan_state['last_scan_ok'],
        'next_scan_time': scan_state['next_scan_time'],
        'classifiers_used': scan_state.get('classifiers_used', []),
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


# --- Activity Summary helpers ---

def _query_counts(conn):
    """Status counts from pending_emails."""
    counts = {'pending_review': 0, 'approved': 0, 'rejected': 0,
              'out_of_scope': 0, 'total_processed': 0,
              'active_conversations': 0, 'response_required_pending': 0,
              'auto_approved': 0}
    try:
        for row in conn.execute("SELECT status, COUNT(*) FROM pending_emails GROUP BY status"):
            counts[row[0]] = row[1]
        counts['total_processed'] = conn.execute(
            "SELECT COUNT(*) FROM processed_messages").fetchone()[0]
        counts['active_conversations'] = conn.execute(
            """SELECT COUNT(DISTINCT conversation_id) FROM pending_emails
               WHERE conversation_id IS NOT NULL AND status = 'pending_review'""").fetchone()[0]
        counts['response_required_pending'] = conn.execute(
            "SELECT COUNT(*) FROM pending_emails WHERE status = 'pending_review' AND response_required = 1"
        ).fetchone()[0]
        try:
            counts['auto_approved'] = conn.execute(
                "SELECT COUNT(*) FROM pending_emails WHERE auto_approved = 1"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            pass
    except sqlite3.OperationalError:
        pass
    return counts


def _query_time_series(conn, today, week_start, month_start):
    """Counts for today / this week / this month."""
    periods = {}
    for label, start in [('today', today), ('this_week', week_start), ('this_month', month_start)]:
        p = {'scanned': 0, 'approved': 0, 'rejected': 0, 'processed': 0}
        try:
            p['scanned'] = conn.execute(
                "SELECT COUNT(*) FROM pending_emails WHERE scanned_at >= ?", (start,)).fetchone()[0]
            p['approved'] = conn.execute(
                "SELECT COUNT(*) FROM pending_emails WHERE status = 'approved' AND decided_at >= ?",
                (start,)).fetchone()[0]
            p['rejected'] = conn.execute(
                "SELECT COUNT(*) FROM pending_emails WHERE status = 'rejected' AND decided_at >= ?",
                (start,)).fetchone()[0]
            p['processed'] = conn.execute(
                "SELECT COUNT(*) FROM processed_messages WHERE processed_at >= ?",
                (start,)).fetchone()[0]
        except sqlite3.OperationalError:
            pass
        periods[label] = p
    return periods


def _query_attention_needed(conn):
    """Lists of emails needing attention (max 10 each) plus total counts."""
    attn = {'high_priority': [], 'stale': [], 'response_required': [], 'low_confidence': [],
            'high_priority_count': 0, 'stale_count': 0, 'response_required_count': 0, 'low_confidence_count': 0}
    fields = "id, subject, sender, sender_name, scanned_at, department, ai_priority, confidence"
    try:
        attn['high_priority_count'] = conn.execute(
            "SELECT COUNT(*) FROM pending_emails WHERE ai_priority = 'high' AND status = 'pending_review'"
        ).fetchone()[0]
        for row in conn.execute(
            f"SELECT {fields} FROM pending_emails WHERE ai_priority = 'high' AND status = 'pending_review' ORDER BY scanned_at ASC LIMIT 10"
        ):
            attn['high_priority'].append(dict(row))

        attn['stale_count'] = conn.execute(
            "SELECT COUNT(*) FROM pending_emails WHERE status = 'pending_review' AND scanned_at < datetime('now', '-2 days')"
        ).fetchone()[0]
        for row in conn.execute(
            f"SELECT {fields} FROM pending_emails WHERE status = 'pending_review' AND scanned_at < datetime('now', '-2 days') ORDER BY scanned_at ASC LIMIT 10"
        ):
            attn['stale'].append(dict(row))

        attn['response_required_count'] = conn.execute(
            "SELECT COUNT(*) FROM pending_emails WHERE status = 'pending_review' AND response_required = 1"
        ).fetchone()[0]
        for row in conn.execute(
            f"SELECT {fields} FROM pending_emails WHERE status = 'pending_review' AND response_required = 1 ORDER BY scanned_at ASC LIMIT 10"
        ):
            attn['response_required'].append(dict(row))

        attn['low_confidence_count'] = conn.execute(
            "SELECT COUNT(*) FROM pending_emails WHERE status = 'pending_review' AND confidence IS NOT NULL AND confidence < 0.5"
        ).fetchone()[0]
        for row in conn.execute(
            f"SELECT {fields} FROM pending_emails WHERE status = 'pending_review' AND confidence IS NOT NULL AND confidence < 0.5 ORDER BY confidence ASC LIMIT 10"
        ):
            attn['low_confidence'].append(dict(row))
    except sqlite3.OperationalError:
        pass
    return attn


def _query_breakdowns(conn):
    """Department and doc_type breakdowns."""
    by_dept = []
    by_type = []
    try:
        for row in conn.execute(
            """SELECT COALESCE(department, 'Unassigned') as dept, status, COUNT(*) as cnt
               FROM pending_emails WHERE status IN ('pending_review', 'approved')
               GROUP BY dept, status ORDER BY cnt DESC"""
        ):
            by_dept.append(dict(row))
        for row in conn.execute(
            """SELECT COALESCE(doc_type, 'Unknown') as dtype, status, COUNT(*) as cnt
               FROM pending_emails WHERE status IN ('pending_review', 'approved')
               GROUP BY dtype, status ORDER BY cnt DESC"""
        ):
            by_type.append(dict(row))
    except sqlite3.OperationalError:
        pass
    # Aggregate into sorted list by total descending
    def _agg(rows, key):
        result = {}
        for r in rows:
            name = r[key]
            if name not in result:
                result[name] = {'pending': 0, 'processed': 0}
            if r['status'] == 'pending_review':
                result[name]['pending'] += r['cnt']
            else:
                result[name]['processed'] += r['cnt']
        items = [{'name': k, **v} for k, v in result.items()]
        items.sort(key=lambda x: x['pending'] + x['processed'], reverse=True)
        return items
    return {'by_department': _agg(by_dept, 'dept'), 'by_doc_type': _agg(by_type, 'dtype')}


def _query_recent_activity(conn):
    """Last 15 actions across approvals, rejections, and new scans."""
    items = []
    try:
        for row in conn.execute(
            """SELECT id, subject, sender_name, decided_at as time, 'approved' as action
               FROM pending_emails WHERE status = 'approved' AND decided_at IS NOT NULL
               ORDER BY decided_at DESC LIMIT 15"""
        ):
            items.append(dict(row))
        for row in conn.execute(
            """SELECT id, subject, sender_name, decided_at as time, 'rejected' as action
               FROM pending_emails WHERE status = 'rejected' AND decided_at IS NOT NULL
               ORDER BY decided_at DESC LIMIT 15"""
        ):
            items.append(dict(row))
        for row in conn.execute(
            """SELECT id, subject, sender_name, scanned_at as time, 'scanned' as action
               FROM pending_emails WHERE status = 'pending_review'
               ORDER BY scanned_at DESC LIMIT 15"""
        ):
            items.append(dict(row))
    except sqlite3.OperationalError:
        pass
    items.sort(key=lambda x: x.get('time') or '', reverse=True)
    return items[:15]


def _query_top_senders(conn):
    """Top 10 senders by total volume."""
    senders = {}
    try:
        for row in conn.execute(
            """SELECT sender, COUNT(*) as cnt FROM pending_emails
               WHERE sender IS NOT NULL AND sender != ''
               GROUP BY sender ORDER BY cnt DESC LIMIT 10"""
        ):
            senders[row['sender']] = row['cnt']
        for row in conn.execute(
            """SELECT sender, COUNT(*) as cnt FROM processed_messages
               WHERE sender IS NOT NULL AND sender != ''
               GROUP BY sender ORDER BY cnt DESC LIMIT 10"""
        ):
            s = row['sender']
            senders[s] = senders.get(s, 0) + row['cnt']
    except sqlite3.OperationalError:
        pass
    ranked = sorted(senders.items(), key=lambda x: x[1], reverse=True)[:10]
    return [{'sender': s, 'count': c} for s, c in ranked]


def _query_classifier_performance(conn):
    """Classifier method breakdown with accuracy stats."""
    perfs = []
    try:
        for row in conn.execute(
            """SELECT classifier_method,
                      COUNT(*) as total,
                      AVG(confidence) as avg_conf,
                      SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved,
                      SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                      SUM(CASE WHEN status = 'out_of_scope' THEN 1 ELSE 0 END) as out_of_scope
               FROM pending_emails
               WHERE classifier_method IS NOT NULL AND classifier_method != ''
               GROUP BY classifier_method ORDER BY total DESC"""
        ):
            perfs.append({
                'method': row['classifier_method'],
                'total': row['total'],
                'avg_confidence': round(row['avg_conf'], 2) if row['avg_conf'] else 0,
                'approved': row['approved'],
                'rejected': row['rejected'],
                'out_of_scope': row['out_of_scope'],
            })
    except sqlite3.OperationalError:
        pass
    return perfs


def _empty_activity_summary():
    """Fallback with zeros when no DB exists."""
    return {
        'counts': {'pending_review': 0, 'approved': 0, 'rejected': 0,
                    'out_of_scope': 0, 'total_processed': 0,
                    'active_conversations': 0, 'response_required_pending': 0,
                    'auto_approved': 0},
        'time_series': {
            'today': {'scanned': 0, 'approved': 0, 'rejected': 0, 'processed': 0},
            'this_week': {'scanned': 0, 'approved': 0, 'rejected': 0, 'processed': 0},
            'this_month': {'scanned': 0, 'approved': 0, 'rejected': 0, 'processed': 0},
        },
        'attention': {'high_priority': [], 'stale': [], 'response_required': [], 'low_confidence': [],
                      'high_priority_count': 0, 'stale_count': 0, 'response_required_count': 0, 'low_confidence_count': 0},
        'breakdowns': {'by_department': [], 'by_doc_type': []},
        'recent_activity': [],
        'top_senders': [],
        'classifier_performance': [],
    }


@app.route('/api/activity-summary')
def api_activity_summary():
    """Comprehensive activity summary for the Home tab dashboard."""
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return jsonify(_empty_activity_summary())

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        week_start = (now - timedelta(days=now.weekday())).strftime('%Y-%m-%d')
        month_start = now.strftime('%Y-%m-01')

        result = {
            'counts': _query_counts(conn),
            'time_series': _query_time_series(conn, today, week_start, month_start),
            'attention': _query_attention_needed(conn),
            'breakdowns': _query_breakdowns(conn),
            'recent_activity': _query_recent_activity(conn),
            'top_senders': _query_top_senders(conn),
            'classifier_performance': _query_classifier_performance(conn),
        }
        conn.close()
        return jsonify(result)
    except Exception:
        return jsonify(_empty_activity_summary())


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


@app.route('/api/scan-progress')
def api_scan_progress():
    """Return real-time scan progress for the UI progress bar."""
    return jsonify(scan_progress)


@app.route('/api/scan', methods=['POST'])
def api_scan():
    if scan_state['status'] == 'scanning':
        return jsonify({'message': 'Scan already in progress'}), 409

    run_scan_in_thread()
    return jsonify({'message': 'Scan started'}), 202


@app.route('/api/scan-all', methods=['POST'])
def api_scan_all():
    """One-time full inbox scan — fetches ALL messages (read + unread), skips already-known."""
    if scan_state['status'] == 'scanning':
        return jsonify({'message': 'Scan already in progress'}), 409

    def _run_full_scan():
        if not scan_lock.acquire(blocking=False):
            return
        try:
            scan_state['status'] = 'scanning'
            _reset_scan_progress()
            _ensure_logging()
            _ensure_backfill()
            logger.info("Dashboard: starting FULL inbox scan")
            from email_interface.scanner import scan_all_emails
            counts = scan_all_emails(base_dir=BASE_DIR, progress_callback=_scan_progress_callback)
            auto_str = f", {counts['auto_assigned']} auto-assigned" if counts.get('auto_assigned') else ''
            inherited_str = f", {counts['inherited']} inherited" if counts.get('inherited') else ''
            scan_state['status'] = 'idle'
            scan_state['last_scan_time'] = datetime.now().isoformat()
            scan_state['last_scan_result'] = (
                f"FULL SCAN: {counts['total']} total, {counts['queued']} queued{auto_str}{inherited_str}, "
                f"{counts['already_known']} already known, "
                f"{counts['out_of_scope']} out-of-scope, {counts['skipped']} skipped"
            )
            scan_state['last_scan_ok'] = counts['errors'] == 0
            scan_progress['active'] = False
            logger.info("Dashboard: full scan complete - %s", scan_state['last_scan_result'])
        except Exception as e:
            scan_state['status'] = 'error'
            scan_state['last_scan_time'] = datetime.now().isoformat()
            scan_state['last_scan_result'] = f"Full scan error: {e}"
            scan_state['last_scan_ok'] = False
            scan_progress['active'] = False
            logger.error("Dashboard: full scan failed - %s", e, exc_info=True)
        finally:
            scan_lock.release()

    t = threading.Thread(target=_run_full_scan, daemon=True)
    t.start()
    return jsonify({'message': 'Full inbox scan started — this may take a few minutes'}), 202


@app.route('/api/emails')
def api_emails():
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return jsonify({'emails': [], 'total': 0, 'today': 0, 'with_attachments': 0,
                        'page': 0, 'pages': 0, 'query': ''})

    query = request.args.get('q', '').strip()
    page = max(0, int(request.args.get('page', 0)))
    per_page = min(200, max(1, int(request.args.get('per_page', 50))))

    tracker = _get_tracker()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        base_where = ""
        join_where = ""
        params = []
        if query:
            base_where = """WHERE (subject LIKE ? OR sender LIKE ?
                            OR sender_name LIKE ? OR transmittal_no LIKE ?)"""
            join_where = """WHERE (pm.subject LIKE ? OR pm.sender LIKE ?
                            OR pm.sender_name LIKE ? OR pm.transmittal_no LIKE ?)"""
            like_q = f'%{query}%'
            params = [like_q, like_q, like_q, like_q]

        # Count matching rows
        count_sql = f"SELECT COUNT(*) FROM processed_messages {base_where}"
        total_matching = conn.execute(count_sql, params).fetchone()[0]
        total_pages = max(1, (total_matching + per_page - 1) // per_page)
        page = min(page, total_pages - 1)

        # Fetch page — get all emails then group by transmittal
        cur = conn.execute(
            f"""SELECT pm.transmittal_no, pm.processed_at, pm.sender, pm.sender_name,
                      pm.to_recipients, pm.cc_recipients, pm.subject, pm.attachment_count,
                      pm.doc_type, pm.discipline, pm.department, pm.response_required,
                      pm.references_json, pm.conversation_id, pm.message_id,
                      pe.classifier_method, pe.auto_approved
               FROM processed_messages pm
               LEFT JOIN pending_emails pe ON pm.message_id = pe.message_id
               {join_where}
               ORDER BY pm.processed_at DESC LIMIT ? OFFSET ?""",
            params + [per_page, page * per_page],
        )
        emails = []
        for row in cur.fetchall():
            e = dict(row)
            contact = tracker.find_contact_by_email(e.get('sender', ''))
            e['sender_company'] = contact['company'] if contact else ''
            e['sender_known'] = contact is not None
            e['thread_count'] = tracker.get_thread_count(e.get('conversation_id')) if e.get('conversation_id') else 0
            emails.append(e)

        # Group emails by transmittal_no — first occurrence is parent, rest are follow-ups
        from collections import OrderedDict
        groups = OrderedDict()
        for e in emails:
            tn = e.get('transmittal_no', '')
            if tn not in groups:
                e['follow_ups'] = []
                groups[tn] = e
            else:
                groups[tn]['follow_ups'].append(e)
        emails = list(groups.values())

        # Stats (unfiltered)
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
            'page': page,
            'pages': total_pages,
            'query': query,
            'total_matching': total_matching,
        })
    except sqlite3.OperationalError:
        return jsonify({'emails': [], 'total': 0, 'today': 0, 'with_attachments': 0,
                        'page': 0, 'pages': 0, 'query': query})
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
        # Enrich with contact info and thread data
        for email in pending:
            contact = tracker.find_contact_by_email(email.get('sender', ''))
            email['sender_company'] = contact['company'] if contact else ''
            email['sender_known'] = contact is not None
            conv_id = email.get('conversation_id')
            email['thread_count'] = tracker.get_thread_count(conv_id) if conv_id else 0
            email['conversation_position'] = email.get('conversation_position', 1)
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

        # Get attachment metadata — only document files
        attachments = tracker.get_pending_attachments_meta(pending_id)
        email['attachments'] = [a for a in attachments if _is_document(a.get('filename', ''))]

        # Add cleaned body summary
        email['body_summary'] = _summarize_body(email.get('body', ''))
        email['has_html_body'] = bool(email.get('body_html'))

        # Add conversation/thread data
        conv_id = email.get('conversation_id')
        if conv_id:
            conv = tracker.get_conversation(conv_id)
            if conv:
                email['conversation'] = conv
                email['thread_emails'] = tracker.get_conversation_emails(conv_id)
                email['thread_count'] = conv.get('email_count', 1)
            else:
                email['thread_count'] = 0
        else:
            email['thread_count'] = 0

        return jsonify(email)
    finally:
        tracker.close()


@app.route('/api/pending/<int:pending_id>/approve', methods=['POST'])
def api_approve(pending_id):
    from email_interface.approval import perform_full_approval
    edits = request.get_json(silent=True) or {}
    tracker = _get_tracker()
    try:
        result = perform_full_approval(tracker, pending_id, BASE_DIR, edits=edits)
        return jsonify({
            'message': f"Approved as {result['transmittal_no']}",
            'transmittal_no': result['transmittal_no'],
            'attachment_folder': result['attachment_folder'],
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


@app.route('/api/pending/<int:pending_id>/reopen', methods=['POST'])
def api_reopen(pending_id):
    tracker = _get_tracker()
    try:
        tracker.reopen_rejected(pending_id)
        return jsonify({'message': 'Reopened for review'})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    finally:
        tracker.close()


@app.route('/api/pending/<int:pending_id>/attachments')
def api_pending_attachments(pending_id):
    """Return document attachment metadata (filenames, sizes), excluding images/signatures."""
    tracker = _get_tracker()
    try:
        attachments = tracker.get_pending_attachments_meta(pending_id)
        docs = [a for a in attachments if _is_document(a.get('filename', ''))]
        return jsonify({'attachments': docs})
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


@app.route('/api/processed/<message_id>/attachments')
def api_processed_attachments(message_id):
    """Return document attachments for a processed email (via pending_emails link)."""
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return jsonify({'attachments': []})
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id FROM pending_emails WHERE message_id = ?", (message_id,)
        ).fetchone()
        if not row:
            return jsonify({'attachments': []})
        pending_id = row['id']
        cur = conn.execute(
            "SELECT id, filename, content_type, size FROM pending_attachments WHERE pending_email_id = ?",
            (pending_id,),
        )
        atts = [dict(r) for r in cur.fetchall() if _is_document(r['filename'])]
        return jsonify({'attachments': atts, 'pending_id': pending_id})
    except sqlite3.OperationalError:
        return jsonify({'attachments': []})
    finally:
        conn.close()


@app.route('/api/pending/bulk-approve', methods=['POST'])
def api_bulk_approve():
    from email_interface.approval import perform_full_approval
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
                result = perform_full_approval(tracker, pid, BASE_DIR)
                results.append({'id': pid, 'transmittal_no': result['transmittal_no']})
            except Exception as e:
                errors.append({'id': pid, 'error': str(e)})
    finally:
        tracker.close()

    return jsonify({'approved': results, 'errors': errors})


@app.route('/api/accuracy')
def api_accuracy():
    """Return per-classifier accuracy stats and common correction patterns."""
    tracker = _get_tracker()
    try:
        stats = tracker.get_accuracy_stats()
        return jsonify(stats)
    finally:
        tracker.close()


@app.route('/api/contacts/suggested')
def api_contacts_suggested():
    """Return senders not in contacts but with 2+ approved emails."""
    tracker = _get_tracker()
    try:
        min_emails = int(request.args.get('min_emails', 1))
        suggested = tracker.get_suggested_contacts(min_emails=min_emails)
        return jsonify({'suggested': suggested})
    finally:
        tracker.close()


@app.route('/api/contacts/suggested/add', methods=['POST'])
def api_contacts_suggested_add():
    """One-click add a suggested contact."""
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip()
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    tracker = _get_tracker()
    try:
        existing = tracker.find_contact_by_email(email)
        if existing:
            return jsonify({'error': f'Contact with email "{email}" already exists'}), 409

        name = data.get('name', '').strip() or email.split('@')[0]
        company = data.get('company', '').strip()
        department = data.get('department', '').strip()

        contact_id = tracker.add_contact(
            name=name,
            email=email,
            company=company,
            department=department,
            notes='Added from suggested contacts',
        )
        return jsonify({'id': contact_id, 'message': f'Contact "{name}" added'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': f'Contact with email "{email}" already exists'}), 409
    finally:
        tracker.close()


@app.route('/api/settings/auto-approve', methods=['GET'])
def api_get_auto_approve_settings():
    """Return all auto-processing settings."""
    tracker = _get_tracker()
    try:
        settings = {
            'auto_approve_enabled': tracker.get_setting('auto_approve_enabled') == 'true',
            'auto_approve_threshold_known': float(tracker.get_setting('auto_approve_threshold_known') or '0.75'),
            'auto_approve_threshold_unknown': float(tracker.get_setting('auto_approve_threshold_unknown') or '0.90'),
            'auto_learn_contacts_enabled': tracker.get_setting('auto_learn_contacts_enabled') != 'false',
            'auto_learn_contacts_threshold': int(tracker.get_setting('auto_learn_contacts_threshold') or '2'),
        }
        return jsonify(settings)
    finally:
        tracker.close()


@app.route('/api/settings/auto-approve', methods=['PUT'])
def api_set_auto_approve_settings():
    """Update auto-processing settings with validation."""
    data = request.get_json(silent=True) or {}
    tracker = _get_tracker()
    try:
        if 'auto_approve_enabled' in data:
            tracker.set_setting('auto_approve_enabled',
                                'true' if data['auto_approve_enabled'] else 'false')

        if 'auto_approve_threshold_known' in data:
            val = float(data['auto_approve_threshold_known'])
            if not 0.5 <= val <= 1.0:
                return jsonify({'error': 'Threshold must be between 0.50 and 1.00'}), 400
            tracker.set_setting('auto_approve_threshold_known', str(val))

        if 'auto_approve_threshold_unknown' in data:
            val = float(data['auto_approve_threshold_unknown'])
            if not 0.5 <= val <= 1.0:
                return jsonify({'error': 'Threshold must be between 0.50 and 1.00'}), 400
            tracker.set_setting('auto_approve_threshold_unknown', str(val))

        if 'auto_learn_contacts_enabled' in data:
            tracker.set_setting('auto_learn_contacts_enabled',
                                'true' if data['auto_learn_contacts_enabled'] else 'false')

        if 'auto_learn_contacts_threshold' in data:
            val = int(data['auto_learn_contacts_threshold'])
            if val < 1:
                return jsonify({'error': 'Threshold must be at least 1'}), 400
            tracker.set_setting('auto_learn_contacts_threshold', str(val))

        return jsonify({'message': 'Settings saved'})
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid value: {e}'}), 400
    finally:
        tracker.close()


@app.route('/api/reclassify', methods=['POST'])
def api_reclassify():
    """Re-run classification on pending_review (and optionally out_of_scope) emails
    using current departments, contacts, and custom rules."""
    if scan_state['status'] == 'scanning':
        return jsonify({'message': 'A scan is already in progress'}), 409

    data = request.get_json(silent=True) or {}
    include_out_of_scope = data.get('include_out_of_scope', False)

    def _run_reclassify():
        if not scan_lock.acquire(blocking=False):
            return
        try:
            scan_state['status'] = 'scanning'
            _ensure_logging()
            rlogger = logging.getLogger(__name__)
            rlogger.info("Dashboard: starting re-classification")

            from email_interface.classifier import create_classifier
            from email_interface.config import load_config, resolve_path

            class_cfg = load_config(resolve_path('config/classification_config.yaml', BASE_DIR))
            tracker = _get_tracker()

            departments = tracker.get_departments(active_only=True)
            contacts = tracker.get_contacts(active_only=True)
            method_override = tracker.get_setting('classifier_method')
            custom_instructions = tracker.get_setting('classifier_instructions')
            classifier = create_classifier(
                class_cfg, departments=departments, method=method_override,
                custom_instructions=custom_instructions, contacts=contacts,
            )

            counts = {'total': 0, 'unchanged': 0, 'moved_to_out_of_scope': 0,
                       'moved_to_review': 0, 'dept_updated': 0, 'errors': 0}

            # Fetch emails to re-classify
            statuses = ['pending_review']
            if include_out_of_scope:
                statuses.append('out_of_scope')

            conn = sqlite3.connect(_get_db_path())
            conn.row_factory = sqlite3.Row
            for status in statuses:
                rows = conn.execute(
                    """SELECT id, message_id, sender, sender_name, subject, body,
                              to_recipients, cc_recipients, attachment_count,
                              doc_type, discipline, department, status
                       FROM pending_emails WHERE status = ?""",
                    (status,)
                ).fetchall()
                counts['total'] += len(rows)

                for i, row in enumerate(rows):
                    # Throttle API calls: 3s between requests to stay within
                    # free-tier rate limits (Gemini: 10/min, Groq: 30/min)
                    if i > 0:
                        time.sleep(3)

                    try:
                        msg_data = {
                            'sender': row['sender'] or '',
                            'sender_name': row['sender_name'] or '',
                            'subject': row['subject'] or '',
                            'body': row['body'] or '',
                            'to': json.loads(row['to_recipients'] or '[]'),
                            'cc': json.loads(row['cc_recipients'] or '[]'),
                            'attachments': [],  # no blobs needed for classification
                        }

                        # Combined scope+classify (single API call)
                        result = classifier.classify_full(msg_data)

                        if not result.in_scope and row['status'] == 'pending_review':
                            # Was in review, now out of scope
                            conn.execute(
                                "UPDATE pending_emails SET status = 'out_of_scope', "
                                "doc_type = '', discipline = '', department = '', "
                                "ai_summary = ?, ai_priority = ?, "
                                "classifier_method = ? WHERE id = ?",
                                (result.summary, result.priority,
                                 getattr(classifier, 'name', ''), row['id']),
                            )
                            counts['moved_to_out_of_scope'] += 1
                            rlogger.info("Re-classify: moved to out_of_scope: [%d] %s",
                                         row['id'], row['subject'])
                            continue

                        if result.in_scope and row['status'] == 'out_of_scope':
                            # Was out of scope, now in scope — move to review
                            conn.execute(
                                """UPDATE pending_emails SET status = 'pending_review',
                                   doc_type = ?, discipline = ?, department = ?,
                                   response_required = ?, references_json = ?,
                                   confidence = ?, classifier_method = ?,
                                   ai_summary = ?, ai_priority = ?
                                   WHERE id = ?""",
                                (result.doc_type, result.discipline, result.department,
                                 int(result.response_required),
                                 json.dumps(result.references),
                                 result.confidence,
                                 getattr(classifier, 'name', ''),
                                 result.summary, result.priority,
                                 row['id']),
                            )
                            counts['moved_to_review'] += 1
                            rlogger.info("Re-classify: moved to pending_review: [%d] %s",
                                         row['id'], row['subject'])
                            continue

                        if result.in_scope and row['status'] == 'pending_review':
                            # Still in scope — re-classify to update department/type/etc
                            old_dept = row['department'] or ''
                            new_dept = result.department or ''
                            changed = (
                                old_dept != new_dept
                                or (row['doc_type'] or '') != result.doc_type
                                or (row['discipline'] or '') != result.discipline
                            )
                            if changed:
                                conn.execute(
                                    """UPDATE pending_emails SET
                                       doc_type = ?, discipline = ?, department = ?,
                                       response_required = ?, references_json = ?,
                                       confidence = ?, classifier_method = ?,
                                       ai_summary = ?, ai_priority = ?
                                       WHERE id = ?""",
                                    (result.doc_type, result.discipline, result.department,
                                     int(result.response_required),
                                     json.dumps(result.references),
                                     result.confidence,
                                     getattr(classifier, 'name', ''),
                                     result.summary, result.priority,
                                     row['id']),
                                )
                                counts['dept_updated'] += 1
                                if old_dept != new_dept:
                                    rlogger.info("Re-classify: dept %s→%s: [%d] %s",
                                                 old_dept or '(none)', new_dept or '(none)',
                                                 row['id'], row['subject'])
                            else:
                                counts['unchanged'] += 1
                    except Exception as e:
                        rlogger.error("Re-classify error for id %d: %s", row['id'], e)
                        counts['errors'] += 1

            conn.commit()
            conn.close()
            tracker.close()

            scan_state['status'] = 'idle'
            scan_state['last_scan_time'] = datetime.now().isoformat()
            parts = [f"{counts['total']} checked"]
            if counts['moved_to_out_of_scope']:
                parts.append(f"{counts['moved_to_out_of_scope']} moved to out-of-scope")
            if counts['moved_to_review']:
                parts.append(f"{counts['moved_to_review']} moved to review")
            if counts['dept_updated']:
                parts.append(f"{counts['dept_updated']} updated")
            if counts['unchanged']:
                parts.append(f"{counts['unchanged']} unchanged")
            scan_state['last_scan_result'] = "RE-CLASSIFY: " + ', '.join(parts)
            scan_state['last_scan_ok'] = counts['errors'] == 0
            rlogger.info("Re-classification complete: %s", scan_state['last_scan_result'])

        except Exception as e:
            scan_state['status'] = 'error'
            scan_state['last_scan_time'] = datetime.now().isoformat()
            scan_state['last_scan_result'] = f"Re-classify error: {e}"
            scan_state['last_scan_ok'] = False
            logging.getLogger(__name__).error("Re-classify failed: %s", e, exc_info=True)
        finally:
            scan_lock.release()

    t = threading.Thread(target=_run_reclassify, daemon=True)
    t.start()
    return jsonify({'message': 'Re-classification started'}), 202


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


# --- Conversations ---

@app.route('/api/conversations/<int:conv_id>')
def api_conversation(conv_id):
    """Return full conversation with all member emails."""
    tracker = _get_tracker()
    try:
        conv = tracker.get_conversation(conv_id)
        if not conv:
            return jsonify({'error': 'Conversation not found'}), 404
        emails = tracker.get_conversation_emails(conv_id)
        # Enrich with contact info
        for email in emails:
            contact = tracker.find_contact_by_email(email.get('sender', ''))
            email['sender_company'] = contact['company'] if contact else ''
            email['sender_known'] = contact is not None
        conv['emails'] = emails
        return jsonify(conv)
    finally:
        tracker.close()


# --- Excel Export ---

@app.route('/api/export/excel')
def api_export_excel():
    """Generate correspondence log as Excel download from SQLite data."""
    from email_monitor import EXCEL_HEADERS

    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return jsonify({'error': 'No data available'}), 404

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            """SELECT pm.transmittal_no, pm.processed_at, pm.sender,
                      pm.to_recipients, pm.cc_recipients, pm.subject,
                      pm.references_json, pm.doc_type, pm.discipline,
                      pm.response_required, pm.attachment_count,
                      pm.message_id, pe.attachment_folder
               FROM processed_messages pm
               LEFT JOIN pending_emails pe ON pm.message_id = pe.message_id
               ORDER BY pm.processed_at DESC"""
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return jsonify({'error': 'No data available'}), 404
    finally:
        conn.close()

    wb = Workbook()
    ws = wb.active
    ws.title = 'Correspondence Log'
    ws.append(EXCEL_HEADERS)

    for row in rows:
        # Parse JSON fields to semicolon-separated strings
        to_list = json.loads(row['to_recipients'] or '[]')
        to_str = '; '.join(
            r.get('email', '') if isinstance(r, dict) else str(r)
            for r in to_list
        )
        cc_list = json.loads(row['cc_recipients'] or '[]')
        cc_str = '; '.join(
            r.get('email', '') if isinstance(r, dict) else str(r)
            for r in cc_list
        )
        refs = json.loads(row['references_json'] or '[]')
        refs_str = '; '.join(refs)

        date_str = row['processed_at'] or ''

        ws.append([
            row['transmittal_no'] or '',
            date_str,
            row['sender'] or '',
            to_str,
            cc_str,
            row['subject'] or '',
            refs_str,
            row['doc_type'] or '',
            row['discipline'] or '',
            'Y' if row['response_required'] else 'N',
            row['attachment_count'] or 0,
            row['attachment_folder'] or '',
            row['message_id'] or '',
        ])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    return Response(
        buf.getvalue(),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={
            'Content-Disposition': 'attachment; filename="correspondence_log.xlsx"',
        },
    )


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

    # One-time backfill: group existing emails into conversations
    if not tracker.get_setting('conversations_backfilled'):
        logger.info("Running one-time conversation backfill...")
        tracker.backfill_conversations()
        tracker.set_setting('conversations_backfilled', '1')
        logger.info("Conversation backfill complete.")

    tracker.close()

    start_scheduler()

    print(f"\n  Document Control Dashboard running at http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
