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

from email_interface.auth import create_auth_handler
from email_interface.classifier import create_classifier
from email_interface.config import load_config, resolve_path
from email_interface.imap_client import ImapEmailClient
from email_interface.message_processor import MessageProcessor
from email_interface.persistence import ProcessingTracker

logger = logging.getLogger(__name__)


def scan_and_queue(base_dir=None):
    """Fetch unread emails, classify, and store in-scope ones as pending.

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

    counts = {'scanned': 0, 'queued': 0, 'skipped': 0, 'out_of_scope': 0, 'errors': 0, 'auto_assigned': 0}

    try:
        imap_client.connect()
        folder = email_cfg.get('folders', {}).get('inbox', 'INBOX')
        imap_client.select_folder(folder)

        msg_ids = imap_client.search_unread()
        if not msg_ids:
            logger.info("No unread messages found")
            return counts

        msg_ids = msg_ids[:max_messages]
        logger.info("Scanning up to %d messages", len(msg_ids))

        for msg_id in msg_ids:
            counts['scanned'] += 1
            try:
                raw_msg = imap_client.fetch_message(msg_id)
                msg_data = processor.parse_message(raw_msg)
                message_id = msg_data.get('message_id', '')

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
                    logger.info("Skip filter: %s - %s", msg_data.get('subject', ''), reason)
                    imap_client.mark_as_read(msg_id)
                    counts['skipped'] += 1
                    continue

                # Scope check
                if not classifier.is_in_scope(msg_data):
                    logger.info("Out of scope: %s", msg_data.get('subject', ''))
                    oos_classification = {
                        'doc_type': '', 'discipline': '', 'department': '',
                        'response_required': False, 'references': [],
                        'confidence': None,
                        'classifier_method': getattr(classifier, 'name', ''),
                    }
                    tracker.store_pending_email(msg_data, oos_classification, status='out_of_scope')
                    imap_client.mark_as_read(msg_id)
                    counts['out_of_scope'] += 1
                    continue

                # Classify
                result = classifier.classify(msg_data)

                # Threading: find or create conversation
                conv_id, conv_pos, existing_transmittal = tracker.find_or_create_conversation(msg_data)
                msg_data['conversation_id'] = conv_id
                msg_data['conversation_position'] = conv_pos

                classification = result.to_dict()
                classification['classifier_method'] = getattr(classifier, 'name', '')

                # Auto-assign follow-ups if conversation already has a transmittal
                if existing_transmittal:
                    pending_id = tracker.store_pending_email(
                        msg_data, classification, status='approved')
                    # Set transmittal on the pending record
                    tracker._conn.execute(
                        "UPDATE pending_emails SET transmittal_no = ? WHERE id = ?",
                        (existing_transmittal, pending_id),
                    )
                    # Copy to processed_messages
                    tracker.mark_processed(message_id, existing_transmittal, {
                        **msg_data,
                        'doc_type': classification.get('doc_type', ''),
                        'discipline': classification.get('discipline', ''),
                        'department': classification.get('department', ''),
                        'response_required': classification.get('response_required', False),
                        'references': classification.get('references', []),
                    })
                    tracker._conn.commit()

                    # Store attachment blobs
                    for att in msg_data.get('attachments', []):
                        tracker.store_pending_attachment(
                            pending_id,
                            att.get('filename', 'unnamed'),
                            att.get('content_type', 'application/octet-stream'),
                            att.get('size', len(att.get('data', b''))),
                            att.get('data', b''),
                        )

                    imap_client.mark_as_read(msg_id)
                    counts['auto_assigned'] += 1
                    logger.info("Auto-assigned follow-up to %s: %s - %s",
                                existing_transmittal, message_id,
                                msg_data.get('subject', ''))
                    continue

                # Store as pending (include classifier metadata)
                pending_id = tracker.store_pending_email(msg_data, classification)

                # Store attachment blobs
                for att in msg_data.get('attachments', []):
                    tracker.store_pending_attachment(
                        pending_id,
                        att.get('filename', 'unnamed'),
                        att.get('content_type', 'application/octet-stream'),
                        att.get('size', len(att.get('data', b''))),
                        att.get('data', b''),
                    )

                # Mark as read in IMAP (don't move to Processed yet)
                imap_client.mark_as_read(msg_id)
                counts['queued'] += 1

                logger.info("Queued: %s - %s [%s/%s]",
                            message_id, msg_data.get('subject', ''),
                            result.doc_type, result.discipline)

            except Exception as e:
                logger.error("Failed to scan message %s: %s", msg_id, e, exc_info=True)
                counts['errors'] += 1

        logger.info("Scan complete: %d scanned, %d queued, %d auto-assigned, %d skipped, %d out-of-scope, %d errors",
                     counts['scanned'], counts['queued'], counts['auto_assigned'],
                     counts['skipped'], counts['out_of_scope'], counts['errors'])

    finally:
        tracker.close()
        imap_client.disconnect()

    return counts


def scan_all_emails(base_dir=None):
    """Fetch ALL emails in inbox (read and unread), skip already-known ones.

    Use this once to backfill the entire mailbox so no emails are lost.
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

    counts = {
        'total': 0, 'scanned': 0, 'queued': 0, 'skipped': 0,
        'out_of_scope': 0, 'errors': 0, 'auto_assigned': 0, 'already_known': 0,
    }

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

        for msg_id in msg_ids:
            counts['scanned'] += 1
            try:
                raw_msg = imap_client.fetch_message(msg_id)
                msg_data = processor.parse_message(raw_msg)
                message_id = msg_data.get('message_id', '')

                # Skip if already in our database
                if tracker.is_processed(message_id) or tracker.is_pending(message_id):
                    counts['already_known'] += 1
                    continue

                # Apply skip filters
                should_process, reason = processor.should_process(msg_data)
                if not should_process:
                    counts['skipped'] += 1
                    continue

                # Scope check
                if not classifier.is_in_scope(msg_data):
                    oos_classification = {
                        'doc_type': '', 'discipline': '', 'department': '',
                        'response_required': False, 'references': [],
                        'confidence': None,
                        'classifier_method': getattr(classifier, 'name', ''),
                    }
                    tracker.store_pending_email(msg_data, oos_classification, status='out_of_scope')
                    counts['out_of_scope'] += 1
                    continue

                # Classify
                result = classifier.classify(msg_data)

                # Threading
                conv_id, conv_pos, existing_transmittal = tracker.find_or_create_conversation(msg_data)
                msg_data['conversation_id'] = conv_id
                msg_data['conversation_position'] = conv_pos

                classification = result.to_dict()
                classification['classifier_method'] = getattr(classifier, 'name', '')

                # Auto-assign follow-ups
                if existing_transmittal:
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

                    for att in msg_data.get('attachments', []):
                        tracker.store_pending_attachment(
                            pending_id,
                            att.get('filename', 'unnamed'),
                            att.get('content_type', 'application/octet-stream'),
                            att.get('size', len(att.get('data', b''))),
                            att.get('data', b''),
                        )

                    counts['auto_assigned'] += 1
                    logger.info("Auto-assigned follow-up to %s: %s",
                                existing_transmittal, msg_data.get('subject', ''))
                    continue

                # Store as pending
                pending_id = tracker.store_pending_email(msg_data, classification)

                for att in msg_data.get('attachments', []):
                    tracker.store_pending_attachment(
                        pending_id,
                        att.get('filename', 'unnamed'),
                        att.get('content_type', 'application/octet-stream'),
                        att.get('size', len(att.get('data', b''))),
                        att.get('data', b''),
                    )

                counts['queued'] += 1
                logger.info("Queued: %s - %s [%s/%s]",
                            message_id, msg_data.get('subject', ''),
                            result.doc_type, result.discipline)

            except Exception as e:
                logger.error("Failed to scan message %s: %s", msg_id, e, exc_info=True)
                counts['errors'] += 1

            # Log progress every 25 messages
            if counts['scanned'] % 25 == 0:
                logger.info("Progress: %d/%d scanned, %d queued, %d already known",
                            counts['scanned'], counts['total'],
                            counts['queued'], counts['already_known'])

        logger.info(
            "Full scan complete: %d total, %d new queued, %d auto-assigned, "
            "%d already known, %d skipped, %d out-of-scope, %d errors",
            counts['total'], counts['queued'], counts['auto_assigned'],
            counts['already_known'], counts['skipped'],
            counts['out_of_scope'], counts['errors'],
        )

    finally:
        tracker.close()
        imap_client.disconnect()

    return counts
