import email
import logging
import re
from email import policy
from email.utils import parseaddr, parsedate_to_datetime

logger = logging.getLogger(__name__)


def normalize_subject(subject):
    """Strip RE:/FW:/FWD: prefixes and normalize whitespace."""
    if not subject:
        return ''
    cleaned = re.sub(r'^(\s*(?:re|fw|fwd)\s*:\s*)+', '', subject, flags=re.IGNORECASE)
    return ' '.join(cleaned.split()).strip()


def _parse_address_list(header_value):
    """Parse a header like 'Name <email>, Name2 <email2>' into a list of dicts."""
    if not header_value:
        return []
    recipients = []
    # Split on commas that aren't inside angle brackets
    parts = re.split(r',(?![^<]*>)', header_value)
    for part in parts:
        name, addr = parseaddr(part.strip())
        if addr:
            recipients.append({'name': name, 'email': addr})
    return recipients


def _mask_email(addr):
    """Mask an email address for privacy: user@example.com -> u***@example.com"""
    if '@' not in addr:
        return addr
    local, domain = addr.rsplit('@', 1)
    if len(local) <= 1:
        masked_local = local
    else:
        masked_local = local[0] + '***'
    return f"{masked_local}@{domain}"


class MessageProcessor:
    """Parse IMAP messages and apply classification logic."""

    def __init__(self, classification_config):
        self.ref_regexes = [re.compile(p) for p in classification_config.get('ref_regexes', [])]
        self.type_keywords = classification_config.get('type_keywords', {})
        self.discipline_keywords = classification_config.get('discipline_keywords', {})
        self.response_phrases = [
            p.lower() for p in classification_config.get('response_required_phrases', [])
        ]
        skip = classification_config.get('skip_filters', {})
        self.skip_sender_domains = skip.get('sender_domains', [])
        self.skip_subject_patterns = [re.compile(p, re.IGNORECASE) for p in skip.get('subject_patterns', [])]

    def parse_message(self, raw_bytes):
        """Parse raw RFC822 bytes into a structured dict.

        Returns: {
            message_id, sender, sender_name, to, cc, subject, date, body,
            attachments: [{filename, data, content_type, size}]
        }
        """
        msg = email.message_from_bytes(raw_bytes, policy=policy.default)

        sender_name, sender_email = parseaddr(msg.get('From', ''))
        date = None
        date_str = msg.get('Date', '')
        if date_str:
            try:
                date = parsedate_to_datetime(date_str)
            except Exception:
                date = None

        body = ''
        body_html = ''
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get('Content-Disposition', ''))

                if 'attachment' in disposition or part.get_filename():
                    att_data = part.get_payload(decode=True)
                    if att_data:
                        attachments.append({
                            'filename': part.get_filename() or 'unnamed',
                            'data': att_data,
                            'content_type': content_type,
                            'size': len(att_data),
                        })
                elif content_type == 'text/plain' and not body:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode('utf-8', errors='replace')
                elif content_type == 'text/html' and not body_html:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_html = payload.decode('utf-8', errors='replace')
                        if not body:
                            body = body_html
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                ct = msg.get_content_type()
                decoded = payload.decode('utf-8', errors='replace')
                if ct == 'text/html':
                    body_html = decoded
                body = decoded

        return {
            'message_id': msg.get('Message-ID', ''),
            'sender': sender_email,
            'sender_name': sender_name,
            'to': _parse_address_list(msg.get('To', '')),
            'cc': _parse_address_list(msg.get('Cc', '')),
            'subject': msg.get('Subject', ''),
            'date': date,
            'body': body,
            'body_html': body_html,
            'attachments': attachments,
            'in_reply_to': msg.get('In-Reply-To', '').strip(),
            'email_references': msg.get('References', '').strip(),
        }

    def should_process(self, msg_data):
        """Decide whether to process or skip this email.

        Returns: (bool, reason_string)
        """
        sender = msg_data.get('sender', '').lower()
        for pattern in self.skip_sender_domains:
            if pattern.lower() in sender:
                return False, f"Sender matches skip filter: {pattern}"

        subject = msg_data.get('subject', '')
        for pattern in self.skip_subject_patterns:
            if pattern.search(subject):
                return False, f"Subject matches skip filter: {pattern.pattern}"

        return True, "OK"

    def extract_reference(self, msg_data):
        """Extract document reference numbers from subject and body."""
        text = f"{msg_data.get('subject', '')} {msg_data.get('body', '')}"
        refs = []
        for regex in self.ref_regexes:
            matches = regex.findall(text)
            refs.extend(matches)
        return refs

    def classify_type(self, msg_data):
        """Classify email type based on keyword matching."""
        text = f"{msg_data.get('subject', '')} {msg_data.get('body', '')}".lower()
        for doc_type, keywords in self.type_keywords.items():
            for kw in keywords:
                if kw.lower() in text:
                    return doc_type
        return 'Others'

    def classify_discipline(self, msg_data):
        """Classify discipline based on keyword matching."""
        text = f"{msg_data.get('subject', '')} {msg_data.get('body', '')}".lower()
        for discipline, keywords in self.discipline_keywords.items():
            for kw in keywords:
                if kw.lower() in text:
                    return discipline
        return 'General'

    def detect_response_required(self, msg_data):
        """Check if the email requires a response."""
        text = f"{msg_data.get('subject', '')} {msg_data.get('body', '')}".lower()
        for phrase in self.response_phrases:
            if phrase in text:
                return True
        return False

    def mask_emails_in_text(self, text):
        """Mask all email addresses found in text."""
        email_pattern = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
        return email_pattern.sub(lambda m: _mask_email(m.group()), text)
