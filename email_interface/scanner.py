"""Phase 1: Scan mailbox and queue emails for review.

Reuses existing auth, imap_client, message_processor components.
Stores in-scope emails as pending in SQLite (with attachment blobs)
instead of immediately assigning transmittal numbers.

scan_and_queue()    — normal mode: fetches UNREAD emails only
scan_all_emails()   — full inbox scan: fetches ALL emails, skips already-known ones
"""

import logging
import os
from datetime import datetime

from email_interface.approval import perform_full_approval
from email_interface.auth import create_auth_handler
from email_interface.classifier import create_classifier
from email_interface.config import load_config, resolve_path
from email_interface.imap_client import ImapEmailClient
from email_interface.message_processor import MessageProcessor
from email_interface.persistence import ProcessingTracker

logger = logging.getLogger(__name__)

BATCH_SIZE = 5


DOCUMENT_EXTENSIONS = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                       '.dwg', '.dxf', '.msg', '.zip', '.rar', '.7z', '.csv', '.txt'}


def _store_attachments(tracker, pending_id, msg_data):
    """Store document attachment blobs for a pending email (skip images/signatures)."""
    for att in msg_data.get('attachments', []):
        filename = att.get('filename', 'unnamed')
        ext = os.path.splitext(filename.lower())[1]
        if ext and ext not in DOCUMENT_EXTENSIONS:
            logger.debug("Skipping non-document attachment: %s", filename)
            continue
        tracker.store_pending_attachment(
            pending_id,
            filename,
            att.get('content_type', 'application/octet-stream'),
            att.get('size', len(att.get('data', b''))),
            att.get('data', b''),
        )


def _auto_assign(tracker, msg_data, classification, existing_transmittal):
    """Auto-assign a follow-up email to its existing transmittal."""
    message_id = msg_data.get('message_id', '')
    pending_id = tracker.store_pending_email(
        msg_data, classification, status='approved')
    tracker._conn.execute(
        "UPDATE pending_emails SET transmittal_no = ? WHERE id = ?",
        (existing_transmittal, pending_id),
    )
    tracker.mark_processed(message_id, existing_transmittal, {
        **msg_data,
        'doc_type': classification.get('doc_type', ''),
        'discipline': classification.get('discipline', ''),
        'department': classification.get('department', ''),
        'response_required': classification.get('response_required', False),
        'references': classification.get('references', []),
    })
    tracker._conn.commit()
    _store_attachments(tracker, pending_id, msg_data)
    return pending_id


def _should_auto_approve(tracker, result, contacts_list):
    """Decide whether to auto-approve an email based on confidence and sender trust.

    Args:
        tracker: ProcessingTracker instance
        result: ClassificationResult from classifier
        contacts_list: List of active contact dicts

    Returns:
        True if the email should be auto-approved.
    """
    enabled = tracker.get_setting('auto_approve_enabled')
    if enabled != 'true':
        return False

    confidence = result.confidence or 0

    # Check if sender is a known contact
    sender = getattr(result, '_sender', '') or ''
    contact_emails = {c['email'].lower() for c in contacts_list}
    is_known = sender.lower().strip() in contact_emails

    if is_known:
        threshold = float(tracker.get_setting('auto_approve_threshold_known') or '0.75')
    else:
        threshold = float(tracker.get_setting('auto_approve_threshold_unknown') or '0.90')

    return confidence >= threshold


def scan_and_queue(base_dir=None, progress_callback=None):
    """Fetch unread emails, classify, and store in-scope ones as pending.

    Args:
        base_dir: Project base directory
        progress_callback: Optional function(counts, current_subject) for progress updates

    Returns: dict with counts {scanned, queued, skipped, out_of_scope, errors}
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load configs
    email_cfg = load_config(resolve_path('config/email_config.yaml', base_dir))
    class_cfg = load_config(resolve_path('config/classification_config.yaml', base_dir))

    # Initialize components
    auth = create_auth_handler(email_cfg['auth'], email_cfg['mailbox']['email_address'])
    imap_client = ImapEmailClient(
        auth_handler=auth,
        host=email_cfg['server']['imap_host'],
        port=email_cfg['server'].get('imap_port', 993),
    )
    processor = MessageProcessor(class_cfg)

    db_path = resolve_path('data/tracking.db', base_dir)
    prefix = class_cfg.get('transmittal', {}).get('prefix', 'TRN')
    tracker = ProcessingTracker(db_path=db_path, prefix=prefix)

    # Seed departments from YAML on first run
    tracker.seed_default_departments(class_cfg.get('discipline_keywords', {}))

    # Create classifier with departments + contacts from SQLite + optional dashboard override
    departments = tracker.get_departments(active_only=True)
    contacts = tracker.get_contacts(active_only=True)
    method_override = tracker.get_setting('classifier_method')
    custom_instructions = tracker.get_setting('classifier_instructions')
    classifier = create_classifier(
        class_cfg, departments=departments, method=method_override,
        custom_instructions=custom_instructions, contacts=contacts,
    )

    max_messages = email_cfg.get('polling', {}).get('max_messages_per_run', 50)
    all_in_scope = class_cfg.get('all_in_scope', False)
    if all_in_scope:
        logger.info("Forwarded mailbox mode: all emails treated as in-scope")

    counts = {'scanned': 0, 'queued': 0, 'skipped': 0, 'out_of_scope': 0,
              'errors': 0, 'auto_assigned': 0, 'inherited': 0, 'total': 0,
              'auto_approved': 0}

    def _progress(subject='', phase='Scanning'):
        if progress_callback:
            progress_callback(counts, subject, phase)

    try:
        imap_client.connect()
        folder = email_cfg.get('folders', {}).get('inbox', 'INBOX')
        imap_client.select_folder(folder)

        msg_ids = imap_client.search_unread()
        if not msg_ids:
            logger.info("No unread messages found")
            return counts

        msg_ids = msg_ids[:max_messages]
        counts['total'] = len(msg_ids)
        logger.info("Scanning up to %d messages", len(msg_ids))

        for msg_id in msg_ids:
            counts['scanned'] += 1
            try:
                raw_msg = imap_client.fetch_message(msg_id)
                msg_data = processor.parse_message(raw_msg)
                message_id = msg_data.get('message_id', '')
                subject = msg_data.get('subject', '')

                _progress(subject)

                # Skip if already processed or already pending
                if tracker.is_processed(message_id):
                    logger.info("Skipping already-processed: %s", message_id)
                    imap_client.mark_as_read(msg_id)
                    counts['skipped'] += 1
                    continue

                if tracker.is_pending(message_id):
                    logger.info("Skipping already-pending: %s", message_id)
                    imap_client.mark_as_read(msg_id)
                    counts['skipped'] += 1
                    continue

                # Apply skip filters (noreply, out-of-office, etc.)
                should_process, reason = processor.should_process(msg_data)
                if not should_process:
                    logger.info("Skip filter: %s - %s", subject, reason)
                    imap_client.mark_as_read(msg_id)
                    counts['skipped'] += 1
                    continue

                # Threading: find or create conversation BEFORE classification
                conv_id, conv_pos, existing_transmittal = tracker.find_or_create_conversation(msg_data)
                msg_data['conversation_id'] = conv_id
                msg_data['conversation_position'] = conv_pos

                # Auto-assign follow-ups if conversation already has a transmittal
                if existing_transmittal:
                    classification = {
                        'doc_type': '', 'discipline': '', 'department': '',
                        'response_required': False, 'references': [],
                        'confidence': 1.0, 'summary': f'Follow-up auto-assigned to {existing_transmittal}',
                        'priority': 'medium',
                        'classifier_method': 'Auto-assigned',
                    }
                    _auto_assign(tracker, msg_data, classification, existing_transmittal)
                    imap_client.mark_as_read(msg_id)
                    counts['auto_assigned'] += 1
                    logger.info("Auto-assigned follow-up to %s: %s - %s",
                                existing_transmittal, message_id, subject)
                    _progress(subject)
                    continue

                # Feature 3: Inherit parent classification for follow-ups
                if conv_pos > 1:
                    parent_cls = tracker.get_conversation_classification(conv_id)
                    if parent_cls:
                        parent_cls['classifier_method'] = 'Inherited'
                        pending_id = tracker.store_pending_email(msg_data, parent_cls)
                        _store_attachments(tracker, pending_id, msg_data)
                        imap_client.mark_as_read(msg_id)
                        counts['inherited'] += 1
                        counts['queued'] += 1
                        logger.info("Inherited classification for follow-up: %s - %s",
                                    message_id, subject)
                        _progress(subject)
                        continue

                # Combined scope+classify (Feature 1: single API call)
                result = classifier.classify_full(msg_data)

                # Forwarded mailbox mode: override scope to always in-scope
                if all_in_scope and not result.in_scope:
                    logger.info("Overriding out-of-scope (all_in_scope=true): %s", subject)
                    result.in_scope = True

                if not result.in_scope:
                    logger.info("Out of scope: %s", subject)
                    oos_classification = {
                        'doc_type': '', 'discipline': '', 'department': '',
                        'response_required': False, 'references': [],
                        'confidence': result.confidence,
                        'summary': result.summary, 'priority': result.priority,
                        'classifier_method': getattr(classifier, 'name', ''),
                    }
                    tracker.store_pending_email(msg_data, oos_classification, status='out_of_scope')
                    imap_client.mark_as_read(msg_id)
                    counts['out_of_scope'] += 1
                    _progress(subject)
                    continue

                classification = result.to_dict()
                classifier_name = getattr(classifier, 'name', '')
                classification['classifier_method'] = classifier_name

                # Check for auto-approve
                result._sender = msg_data.get('sender', '')
                if _should_auto_approve(tracker, result, contacts):
                    classification['classifier_method'] = f"Auto ({classifier_name})"
                    pending_id = tracker.store_pending_email(msg_data, classification)
                    _store_attachments(tracker, pending_id, msg_data)
                    tracker._conn.execute(
                        "UPDATE pending_emails SET auto_approved = 1 WHERE id = ?",
                        (pending_id,))
                    tracker._conn.commit()
                    perform_full_approval(tracker, pending_id, base_dir)
                    imap_client.mark_as_read(msg_id)
                    counts['auto_approved'] += 1
                    counts['queued'] += 1
                    _progress(subject)
                    logger.info("Auto-approved: %s - %s [%s/%s] (conf=%.2f)",
                                message_id, subject,
                                result.doc_type, result.discipline,
                                result.confidence or 0)
                    continue

                # Store as pending
                pending_id = tracker.store_pending_email(msg_data, classification)
                _store_attachments(tracker, pending_id, msg_data)

                # Mark as read in IMAP
                imap_client.mark_as_read(msg_id)
                counts['queued'] += 1
                _progress(subject)

                logger.info("Queued: %s - %s [%s/%s]",
                            message_id, subject,
                            result.doc_type, result.discipline)

            except Exception as e:
                logger.error("Failed to scan message %s: %s", msg_id, e, exc_info=True)
                counts['errors'] += 1
                _progress('')

        logger.info("Scan complete: %d scanned, %d queued, %d auto-approved, %d inherited, "
                     "%d auto-assigned, %d skipped, %d out-of-scope, %d errors",
                     counts['scanned'], counts['queued'], counts['auto_approved'],
                     counts['inherited'], counts['auto_assigned'], counts['skipped'],
                     counts['out_of_scope'], counts['errors'])

    finally:
        tracker.close()
        imap_client.disconnect()

    return counts


def scan_all_emails(base_dir=None, progress_callback=None):
    """Fetch ALL emails in inbox (read and unread), skip already-known ones.

    Uses a two-phase approach:
      Phase 1: Fetch & filter all messages into buckets
      Phase 2: Store inherited classifications (no API needed)
      Phase 3: Batch classify remaining emails

    Args:
        base_dir: Project base directory
        progress_callback: Optional function(counts, current_subject) for progress updates

    Returns: dict with counts
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    email_cfg = load_config(resolve_path('config/email_config.yaml', base_dir))
    class_cfg = load_config(resolve_path('config/classification_config.yaml', base_dir))

    auth = create_auth_handler(email_cfg['auth'], email_cfg['mailbox']['email_address'])
    imap_client = ImapEmailClient(
        auth_handler=auth,
        host=email_cfg['server']['imap_host'],
        port=email_cfg['server'].get('imap_port', 993),
    )
    processor = MessageProcessor(class_cfg)

    db_path = resolve_path('data/tracking.db', base_dir)
    prefix = class_cfg.get('transmittal', {}).get('prefix', 'TRN')
    tracker = ProcessingTracker(db_path=db_path, prefix=prefix)

    tracker.seed_default_departments(class_cfg.get('discipline_keywords', {}))

    departments = tracker.get_departments(active_only=True)
    contacts = tracker.get_contacts(active_only=True)
    method_override = tracker.get_setting('classifier_method')
    custom_instructions = tracker.get_setting('classifier_instructions')
    classifier = create_classifier(
        class_cfg, departments=departments, method=method_override,
        custom_instructions=custom_instructions, contacts=contacts,
    )

    all_in_scope = class_cfg.get('all_in_scope', False)

    counts = {
        'total': 0, 'scanned': 0, 'queued': 0, 'skipped': 0,
        'out_of_scope': 0, 'errors': 0, 'auto_assigned': 0,
        'already_known': 0, 'inherited': 0, 'auto_approved': 0,
    }

    def _progress(subject='', phase=''):
        if progress_callback:
            progress_callback(counts, subject, phase)

    try:
        imap_client.connect()
        folder = email_cfg.get('folders', {}).get('inbox', 'INBOX')
        imap_client.select_folder(folder)

        msg_ids = imap_client.search_all()
        counts['total'] = len(msg_ids)
        if not msg_ids:
            logger.info("No messages found in inbox")
            return counts

        logger.info("Full inbox scan: %d total messages", len(msg_ids))

        # Phase 1: Fetch, filter, and bucket
        needs_classification = []  # list of (msg_data, conv_id, conv_pos)

        _progress('', 'Fetching emails')

        for msg_id in msg_ids:
            counts['scanned'] += 1
            try:
                raw_msg = imap_client.fetch_message(msg_id)
                msg_data = processor.parse_message(raw_msg)
                message_id = msg_data.get('message_id', '')
                subject = msg_data.get('subject', '')

                _progress(subject, 'Fetching emails')

                # Skip if already in our database
                if tracker.is_processed(message_id) or tracker.is_pending(message_id):
                    counts['already_known'] += 1
                    continue

                # Apply skip filters
                should_process, reason = processor.should_process(msg_data)
                if not should_process:
                    counts['skipped'] += 1
                    continue

                # Threading: find or create conversation BEFORE classification
                conv_id, conv_pos, existing_transmittal = tracker.find_or_create_conversation(msg_data)
                msg_data['conversation_id'] = conv_id
                msg_data['conversation_position'] = conv_pos

                # Bucket 1: Auto-assign follow-ups with existing transmittal
                if existing_transmittal:
                    classification = {
                        'doc_type': '', 'discipline': '', 'department': '',
                        'response_required': False, 'references': [],
                        'confidence': 1.0, 'summary': f'Follow-up auto-assigned to {existing_transmittal}',
                        'priority': 'medium',
                        'classifier_method': 'Auto-assigned',
                    }
                    _auto_assign(tracker, msg_data, classification, existing_transmittal)
                    counts['auto_assigned'] += 1
                    logger.info("Auto-assigned follow-up to %s: %s",
                                existing_transmittal, subject)
                    continue

                # Bucket 2: Inherit parent classification for follow-ups
                if conv_pos > 1:
                    parent_cls = tracker.get_conversation_classification(conv_id)
                    if parent_cls:
                        parent_cls['classifier_method'] = 'Inherited'
                        pending_id = tracker.store_pending_email(msg_data, parent_cls)
                        _store_attachments(tracker, pending_id, msg_data)
                        counts['inherited'] += 1
                        counts['queued'] += 1
                        logger.info("Inherited classification for follow-up: %s", subject)
                        continue

                # Bucket 3: Needs AI classification
                needs_classification.append(msg_data)

            except Exception as e:
                logger.error("Failed to fetch message %s: %s", msg_id, e, exc_info=True)
                counts['errors'] += 1

            # Log progress every 25 messages
            if counts['scanned'] % 25 == 0:
                logger.info("Phase 1 progress: %d/%d fetched, %d need classification",
                            counts['scanned'], counts['total'], len(needs_classification))

        # Phase 3: Batch classify remaining emails
        logger.info("Phase 3: Batch classifying %d emails in batches of %d",
                     len(needs_classification), BATCH_SIZE)
        _progress('', 'Classifying emails')

        for batch_start in range(0, len(needs_classification), BATCH_SIZE):
            batch = needs_classification[batch_start:batch_start + BATCH_SIZE]
            batch_subjects = [m.get('subject', '')[:50] for m in batch]
            _progress(', '.join(batch_subjects), 'Classifying emails')

            try:
                if len(batch) == 1:
                    results = [classifier.classify_full(batch[0])]
                else:
                    results = classifier.classify_batch(batch)
            except Exception as exc:
                logger.error("Batch classify failed: %s", exc, exc_info=True)
                # Fall back to individual classify_full
                results = []
                for msg in batch:
                    try:
                        results.append(classifier.classify_full(msg))
                    except Exception as e2:
                        logger.error("Individual classify_full also failed: %s", e2)
                        results.append(None)

            for msg_data, result in zip(batch, results):
                try:
                    if result is None:
                        counts['errors'] += 1
                        continue

                    # Forwarded mailbox mode: override scope
                    if all_in_scope and not result.in_scope:
                        result.in_scope = True

                    if not result.in_scope:
                        oos_classification = {
                            'doc_type': '', 'discipline': '', 'department': '',
                            'response_required': False, 'references': [],
                            'confidence': result.confidence,
                            'summary': result.summary, 'priority': result.priority,
                            'classifier_method': getattr(classifier, 'name', ''),
                        }
                        tracker.store_pending_email(msg_data, oos_classification, status='out_of_scope')
                        counts['out_of_scope'] += 1
                        continue

                    classification = result.to_dict()
                    classifier_name = getattr(classifier, 'name', '')
                    classification['classifier_method'] = classifier_name

                    # Check for auto-approve
                    result._sender = msg_data.get('sender', '')
                    if _should_auto_approve(tracker, result, contacts):
                        classification['classifier_method'] = f"Auto ({classifier_name})"
                        pending_id = tracker.store_pending_email(msg_data, classification)
                        _store_attachments(tracker, pending_id, msg_data)
                        tracker._conn.execute(
                            "UPDATE pending_emails SET auto_approved = 1 WHERE id = ?",
                            (pending_id,))
                        tracker._conn.commit()
                        perform_full_approval(tracker, pending_id, base_dir)
                        counts['auto_approved'] += 1
                        counts['queued'] += 1
                        logger.info("Auto-approved: %s [%s/%s] (conf=%.2f)",
                                    msg_data.get('subject', ''),
                                    result.doc_type, result.discipline,
                                    result.confidence or 0)
                        continue

                    pending_id = tracker.store_pending_email(msg_data, classification)
                    _store_attachments(tracker, pending_id, msg_data)
                    counts['queued'] += 1

                    logger.info("Queued: %s [%s/%s]",
                                msg_data.get('subject', ''),
                                result.doc_type, result.discipline)
                except Exception as e:
                    logger.error("Failed to store classified email: %s", e, exc_info=True)
                    counts['errors'] += 1

            logger.info("Batch %d-%d complete: %d/%d classified",
                        batch_start + 1, min(batch_start + BATCH_SIZE, len(needs_classification)),
                        counts['queued'] + counts['out_of_scope'],
                        len(needs_classification))

        logger.info(
            "Full scan complete: %d total, %d new queued, %d auto-approved, %d inherited, "
            "%d auto-assigned, %d already known, %d skipped, %d out-of-scope, %d errors",
            counts['total'], counts['queued'], counts['auto_approved'],
            counts['inherited'], counts['auto_assigned'], counts['already_known'],
            counts['skipped'], counts['out_of_scope'], counts['errors'],
        )

    finally:
        tracker.close()
        imap_client.disconnect()

    return counts
