"""Shared approval workflow used by both dashboard.py and scanner.py.

Extracts the repeated approval logic (save attachments, log to Excel,
detect corrections, auto-learn contacts) into a single reusable function.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)


def _extract_company_from_email(email_addr):
    """Heuristic: extract company name from email domain.

    Examples:
        john@acme.com       -> Acme
        jane@aecom.co.uk    -> Aecom
        user@sub.company.io -> Company
    """
    if not email_addr or '@' not in email_addr:
        return ''
    domain = email_addr.split('@')[1].lower()
    # Remove common TLDs
    parts = domain.split('.')
    # For domains like company.com, company.co.uk, sub.company.io
    # Take the second-to-last part (or first part for 2-part domains)
    if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'ac'):
        name = parts[-3]
    elif len(parts) >= 2:
        name = parts[-2]
    else:
        name = parts[0]
    # Skip generic providers
    generic = {'gmail', 'yahoo', 'hotmail', 'outlook', 'live', 'aol', 'icloud', 'mail', 'protonmail'}
    if name in generic:
        return ''
    return name.capitalize()


def _detect_corrections(tracker, pending_id, edits):
    """Compare user edits against stored values to detect corrections.

    Returns JSON string of corrections or None if no changes.
    """
    if not edits:
        return None

    pe = tracker.get_pending_email(pending_id)
    if not pe:
        return None

    corrections = {}
    for field in ('doc_type', 'discipline', 'department'):
        if field in edits and edits[field] is not None:
            original = pe.get(field, '') or ''
            new_val = edits[field] or ''
            if original != new_val:
                corrections[field] = [original, new_val]

    if corrections:
        return json.dumps(corrections)
    return None


def _save_attachments_to_disk(tracker, pending_id, result, base_dir):
    """Save attachment blobs to disk with signature filtering.

    Returns list of saved file dicts.
    """
    from email_interface.attachment_handler import AttachmentHandler
    from email_interface.config import load_config, resolve_path

    pe = result['pending_email']
    transmittal_no = result['transmittal_no']

    attachments = tracker.get_pending_attachments(pending_id)
    saved_files = []
    if not attachments:
        return saved_files

    email_cfg = load_config(resolve_path('config/email_config.yaml', base_dir))
    att_cfg = email_cfg.get('attachments', {})
    handler = AttachmentHandler(
        base_path=resolve_path(att_cfg.get('base_path', 'Attachments'), base_dir),
        max_size_mb=att_cfg.get('max_size_mb', 25),
        allowed_extensions=att_cfg.get('allowed_extensions'),
        skip_signature_attachments=att_cfg.get('skip_signature_attachments', True),
    )
    for att in attachments:
        if handler.is_signature_attachment(
            att['filename'], att['content_type'],
            att.get('size') or len(att.get('data', b''))
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

    return saved_files


def _update_attachment_folder(tracker, pending_id, saved_files, db_path):
    """Update attachment_folder column on the pending record."""
    if not saved_files:
        return ''
    att_folder = os.path.dirname(saved_files[0]['path'])
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE pending_emails SET attachment_folder = ? WHERE id = ?",
        (att_folder, pending_id),
    )
    conn.commit()
    conn.close()
    return att_folder


def _log_to_excel(pe, transmittal_no, saved_files, base_dir):
    """Append entry to the Excel correspondence log."""
    from email_monitor import update_excel_log
    from email_interface.config import load_config, resolve_path

    email_cfg = load_config(resolve_path('config/email_config.yaml', base_dir))
    excel_path = resolve_path(
        email_cfg.get('excel', {}).get('output_path', 'logs/correspondence_log.xlsx'),
        base_dir,
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


def _maybe_auto_learn_contact(tracker, pe):
    """Auto-create a contact if the sender has enough approved emails.

    Checks the auto_learn_contacts_enabled setting and threshold.
    """
    enabled = tracker.get_setting('auto_learn_contacts_enabled')
    if enabled == 'false':
        return

    sender_email = (pe.get('sender') or '').lower().strip()
    if not sender_email:
        return

    # Skip if already a contact
    existing = tracker.find_contact_by_email(sender_email)
    if existing:
        return

    threshold = int(tracker.get_setting('auto_learn_contacts_threshold') or '2')
    count = tracker.count_approved_from_sender(sender_email)
    if count < threshold:
        return

    # Auto-create the contact
    sender_name = pe.get('sender_name') or ''
    company = _extract_company_from_email(sender_email)
    department = tracker.get_most_common_department_for_sender(sender_email) or ''

    try:
        tracker.add_contact(
            name=sender_name or sender_email.split('@')[0],
            email=sender_email,
            company=company,
            department=department,
            notes='Auto-learned from email history',
        )
        logger.info("Auto-learned contact: %s <%s> [%s/%s]",
                     sender_name, sender_email, company, department)
    except Exception as e:
        # IntegrityError if race condition
        logger.debug("Could not auto-learn contact %s: %s", sender_email, e)


def perform_full_approval(tracker, pending_id, base_dir, edits=None):
    """Complete approval: corrections + transmittal# + save attachments + Excel + learn contacts.

    Args:
        tracker: ProcessingTracker instance
        pending_id: The pending email id to approve
        base_dir: Project base directory
        edits: Optional dict with doc_type/discipline/department overrides

    Returns:
        dict with transmittal_no, attachment_folder, saved_files, corrections
    """
    db_path = os.path.join(base_dir, 'data', 'tracking.db')

    # 1. Detect corrections BEFORE approval changes the values
    corrections = _detect_corrections(tracker, pending_id, edits)

    # 2. DB approval (generates transmittal, copies to processed_messages)
    result = tracker.approve_pending_email(pending_id, edits=edits)
    transmittal_no = result['transmittal_no']
    pe = result['pending_email']

    # 3. Store corrections if any
    if corrections:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE pending_emails SET user_corrections = ? WHERE id = ?",
            (corrections, pending_id),
        )
        conn.commit()
        conn.close()

    # 4. Save attachment blobs to disk
    saved_files = _save_attachments_to_disk(tracker, pending_id, result, base_dir)

    # 5. Update attachment_folder
    att_folder = _update_attachment_folder(tracker, pending_id, saved_files, db_path)

    # 6. Log to Excel
    _log_to_excel(pe, transmittal_no, saved_files, base_dir)

    # 7. Auto-learn contact if threshold met
    _maybe_auto_learn_contact(tracker, pe)

    return {
        'transmittal_no': transmittal_no,
        'attachment_folder': att_folder,
        'saved_files': saved_files,
        'corrections': corrections,
        'auto_approved_siblings': result.get('auto_approved', []),
    }
