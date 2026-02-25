"""SMTP email sending for acknowledgments and reminders.

Uses Python built-in smtplib + email.mime (no new dependencies).
All send functions catch exceptions and return result dicts — approval
never breaks even if SMTP is misconfigured or down.
"""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

# Default templates
DEFAULT_ACK_SUBJECT = 'Receipt Confirmation - {transmittal_no}'
DEFAULT_ACK_BODY = """\
Dear {sender_name},

We acknowledge receipt of your correspondence:

Subject: {subject}
Date Received: {date}
Our Reference: {transmittal_no}

This has been logged and will be processed by our document control team.

Thank you for your submission.

Best regards,
Document Control Department"""

DEFAULT_TEAM_REMINDER_SUBJECT = 'Action Required: {subject} - {transmittal_no}'
DEFAULT_TEAM_REMINDER_BODY = """\
Hi {assigned_to},

This is a reminder that the following item has been assigned to you and requires action:

Subject: {subject}
From: {sender_name}
Date Received: {date}
Reference: {transmittal_no}
Days Since Assignment: {days_since_assigned}

Please action this item at your earliest convenience.

Thank you,
Document Control"""

DEFAULT_REMINDER_SUBJECT = 'Reminder: Response Required - {subject}'
DEFAULT_REMINDER_BODY = """\
Dear {sender_name},

This is a friendly reminder regarding correspondence which requires your response:

Subject: {subject}
Date: {date}
Our Reference: {transmittal_no}
Days Since Receipt: {days_overdue}

Your prompt response would be appreciated.

If you have already responded, please disregard this message.

Best regards,
Document Control Department"""


def _load_smtp_config(base_dir):
    """Load SMTP settings from email_config.yaml + env var overrides."""
    from email_interface.config import load_config, resolve_path
    email_cfg = load_config(resolve_path('config/email_config.yaml', base_dir))
    smtp = email_cfg.get('smtp', {})
    return {
        'enabled': smtp.get('enabled', False),
        'host': os.environ.get('SMTP_HOST', smtp.get('smtp_host', '')),
        'port': int(os.environ.get('SMTP_PORT', smtp.get('smtp_port', 587))),
        'use_tls': os.environ.get('SMTP_USE_TLS', str(smtp.get('use_tls', True))).lower() in ('true', '1', 'yes'),
        'from_name': smtp.get('from_name', 'Document Control'),
    }


def _get_credentials(base_dir):
    """Reuse IMAP credentials for SMTP auth."""
    from email_interface.config import load_config, resolve_path
    email_cfg = load_config(resolve_path('config/email_config.yaml', base_dir))
    auth_cfg = email_cfg.get('auth', {})
    username = os.environ.get('EMAIL_USER', auth_cfg.get('username', ''))
    password = os.environ.get('EMAIL_PASSWORD', auth_cfg.get('password', ''))
    email_address = os.environ.get('EMAIL_ADDRESS',
                                    email_cfg.get('mailbox', {}).get('email_address', username))
    return username, password, email_address


def send_email(base_dir, to_address, subject, body_text, body_html=None, reply_to=None):
    """Send one email via SMTP. NEVER raises — returns {'success': bool, 'error': str|None}."""
    try:
        smtp_cfg = _load_smtp_config(base_dir)
        if not smtp_cfg['host']:
            return {'success': False, 'error': 'SMTP host not configured'}

        username, password, from_address = _get_credentials(base_dir)
        if not from_address:
            return {'success': False, 'error': 'No from address configured'}

        from_header = f"{smtp_cfg['from_name']} <{from_address}>" if smtp_cfg['from_name'] else from_address

        if body_html:
            msg = MIMEMultipart('alternative')
            msg.attach(MIMEText(body_text, 'plain'))
            msg.attach(MIMEText(body_html, 'html'))
        else:
            msg = MIMEText(body_text, 'plain')

        msg['From'] = from_header
        msg['To'] = to_address
        msg['Subject'] = subject
        if reply_to:
            msg['Reply-To'] = reply_to

        if smtp_cfg['use_tls']:
            server = smtplib.SMTP(smtp_cfg['host'], smtp_cfg['port'], timeout=30)
            server.starttls()
        else:
            server = smtplib.SMTP(smtp_cfg['host'], smtp_cfg['port'], timeout=30)

        if username and password:
            server.login(username, password)

        server.sendmail(from_address, [to_address], msg.as_string())
        server.quit()

        logger.info("Email sent to %s: %s", to_address, subject)
        return {'success': True, 'error': None}

    except Exception as e:
        logger.warning("SMTP send failed to %s: %s", to_address, e)
        return {'success': False, 'error': str(e)}


def _render_template(tracker, template_key, default_template, variables):
    """Load template from settings table, substitute variables."""
    template = tracker.get_setting(template_key) or default_template
    try:
        return template.format(**variables)
    except KeyError as e:
        logger.warning("Template variable missing: %s, using default", e)
        return default_template.format(**variables)


def _log_sent_email(tracker, message_id, transmittal_no, recipient,
                    email_type, subject, body, result):
    """Insert into sent_emails table."""
    tracker.log_sent_email(
        message_id=message_id,
        transmittal_no=transmittal_no,
        recipient=recipient,
        email_type=email_type,
        subject=subject,
        body=body,
        status='sent' if result['success'] else 'failed',
        error=result.get('error'),
    )


def send_acknowledgment(base_dir, tracker, pending_email, transmittal_no):
    """Send receipt confirmation. Returns result dict."""
    # Check if ack sending is enabled
    if tracker.get_setting('ack_enabled') != 'true':
        return {'success': False, 'error': 'Acknowledgment sending disabled'}

    # Check SMTP enabled
    if tracker.get_setting('smtp_enabled') != 'true':
        return {'success': False, 'error': 'SMTP disabled'}

    sender = pending_email.get('sender', '')
    if not sender:
        return {'success': False, 'error': 'No sender address'}

    sender_name = pending_email.get('sender_name', '') or sender.split('@')[0]
    email_date = pending_email.get('email_date', '') or ''
    if hasattr(email_date, 'strftime'):
        email_date = email_date.strftime('%Y-%m-%d %H:%M')

    variables = {
        'sender_name': sender_name,
        'subject': pending_email.get('subject', ''),
        'transmittal_no': transmittal_no,
        'date': str(email_date),
    }

    subject = _render_template(tracker, 'ack_template_subject',
                               DEFAULT_ACK_SUBJECT, variables)
    body = _render_template(tracker, 'ack_template_body',
                            DEFAULT_ACK_BODY, variables)

    result = send_email(base_dir, sender, subject, body)

    # Log to sent_emails table
    _log_sent_email(tracker, pending_email.get('message_id', ''),
                    transmittal_no, sender, 'acknowledgment', subject, body, result)

    # Mark acknowledgment sent on records
    if result['success']:
        tracker.mark_acknowledgment_sent(
            pending_email.get('message_id', ''),
            pending_email.get('id'),
        )

    return result


def send_reminder(base_dir, tracker, processed_msg, manual=False):
    """Send response reminder. Returns result dict.

    Args:
        manual: If True, skip the reminder_enabled check (for manual button clicks).
    """
    # Check if reminders are enabled (skip for manual sends)
    if not manual and tracker.get_setting('reminder_enabled') != 'true':
        return {'success': False, 'error': 'Reminders disabled'}

    if tracker.get_setting('smtp_enabled') != 'true':
        return {'success': False, 'error': 'SMTP disabled'}

    sender = processed_msg.get('sender', '')
    if not sender:
        return {'success': False, 'error': 'No sender address'}

    sender_name = processed_msg.get('sender_name', '') or sender.split('@')[0]
    processed_at = processed_msg.get('processed_at', '')

    # Calculate days overdue
    days_overdue = 0
    due_date = processed_msg.get('response_due_date', '')
    if due_date:
        try:
            from dateutil.parser import parse as parse_date
            due_dt = parse_date(due_date)
            days_overdue = (datetime.now() - due_dt).days
        except Exception:
            pass

    variables = {
        'sender_name': sender_name,
        'subject': processed_msg.get('subject', ''),
        'transmittal_no': processed_msg.get('transmittal_no', ''),
        'date': str(processed_at),
        'days_overdue': str(max(days_overdue, 0)),
    }

    subject = _render_template(tracker, 'reminder_template_subject',
                               DEFAULT_REMINDER_SUBJECT, variables)
    body = _render_template(tracker, 'reminder_template_body',
                            DEFAULT_REMINDER_BODY, variables)

    result = send_email(base_dir, sender, subject, body)

    # Log to sent_emails table
    _log_sent_email(tracker, processed_msg.get('message_id', ''),
                    processed_msg.get('transmittal_no', ''),
                    sender, 'reminder', subject, body, result)

    # Update reminder count
    if result['success']:
        tracker.update_reminder_sent(processed_msg.get('message_id', ''))

    return result


def _resolve_team_email(tracker, name):
    """Look up a team member's email by name. Returns email or None."""
    contact = tracker.find_contact_by_name(name)
    return contact['email'] if contact else None


def send_team_reminder(base_dir, tracker, processed_msg, manual=False):
    """Send reminder to an assigned team member. Returns result dict.

    Args:
        manual: If True, skip the team_reminder_enabled check (for nudge button).
    """
    if not manual and tracker.get_setting('team_reminder_enabled') != 'true':
        return {'success': False, 'error': 'Team reminders disabled'}

    if tracker.get_setting('smtp_enabled') != 'true':
        return {'success': False, 'error': 'SMTP disabled'}

    assigned_to = processed_msg.get('assigned_to', '')
    if not assigned_to:
        return {'success': False, 'error': 'No team member assigned'}

    to_address = _resolve_team_email(tracker, assigned_to)
    if not to_address:
        return {'success': False, 'error': f'No email found for team member: {assigned_to}'}

    sender_name = processed_msg.get('sender_name', '') or processed_msg.get('sender', '').split('@')[0]
    processed_at = processed_msg.get('processed_at', '')

    # Calculate days since assignment
    days_since = 0
    if processed_at:
        try:
            from dateutil.parser import parse as parse_date
            proc_dt = parse_date(processed_at)
            days_since = (datetime.now() - proc_dt).days
        except Exception:
            pass

    variables = {
        'assigned_to': assigned_to,
        'sender_name': sender_name,
        'subject': processed_msg.get('subject', ''),
        'transmittal_no': processed_msg.get('transmittal_no', ''),
        'date': str(processed_at),
        'days_since_assigned': str(max(days_since, 0)),
    }

    subject = _render_template(tracker, 'team_reminder_template_subject',
                               DEFAULT_TEAM_REMINDER_SUBJECT, variables)
    body = _render_template(tracker, 'team_reminder_template_body',
                            DEFAULT_TEAM_REMINDER_BODY, variables)

    result = send_email(base_dir, to_address, subject, body)

    # Log to sent_emails table
    _log_sent_email(tracker, processed_msg.get('message_id', ''),
                    processed_msg.get('transmittal_no', ''),
                    to_address, 'team_reminder', subject, body, result)

    # Update team reminder count
    if result['success']:
        tracker.update_team_reminder_sent(processed_msg.get('message_id', ''))

    return result
