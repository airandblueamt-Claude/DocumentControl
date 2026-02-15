import json
import logging
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# Months for folder naming
MONTH_NAMES = [
    '', 'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]

# Patterns that identify signature/logo attachments
_SIGNATURE_FILENAME_PATTERNS = [
    re.compile(r'^image0\d+\.\w+$', re.IGNORECASE),
    re.compile(r'^logo[._\-]', re.IGNORECASE),
    re.compile(r'^signature[._\-]', re.IGNORECASE),
    re.compile(r'^banner[._\-]', re.IGNORECASE),
    re.compile(r'^icon[._\-]', re.IGNORECASE),
    re.compile(r'^linkedin[._\-]', re.IGNORECASE),
    re.compile(r'^facebook[._\-]', re.IGNORECASE),
    re.compile(r'^twitter[._\-]', re.IGNORECASE),
    re.compile(r'^x[._\-]logo', re.IGNORECASE),
    re.compile(r'^instagram[._\-]', re.IGNORECASE),
]

_SIGNATURE_IMAGE_TYPES = {'image/png', 'image/jpeg', 'image/gif', 'image/bmp'}

# Max size for signature images (50 KB)
_SIGNATURE_MAX_SIZE = 50 * 1024


def _sanitize_filename(name):
    """Remove unsafe characters from a filename."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = name.strip('. ')
    return name or 'unnamed_attachment'


class AttachmentHandler:
    """Download and organize email attachments into a date/transmittal folder structure."""

    def __init__(self, base_path, max_size_mb=25, allowed_extensions=None,
                 skip_signature_attachments=True):
        self.base_path = base_path
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.allowed_extensions = allowed_extensions
        self.skip_signature_attachments = skip_signature_attachments

    def _is_signature_attachment(self, filename, content_type, size):
        """Return True if this attachment looks like an email signature image.

        Checks:
        - Small images (< 50KB) with image content types
        - Common signature filenames: image001.png, logo.*, signature.*, etc.
        - Inline CID images matching image0XX pattern
        """
        ct = (content_type or '').lower()
        fn = (filename or '').strip()

        # Must be an image type to be a signature
        if ct not in _SIGNATURE_IMAGE_TYPES:
            return False

        # Check if it's a small image (< 50KB) — likely a signature/logo
        if size is not None and size < _SIGNATURE_MAX_SIZE:
            # Small image — check filename patterns
            for pattern in _SIGNATURE_FILENAME_PATTERNS:
                if pattern.match(fn):
                    return True

        # Even for larger images, flag common signature names
        for pattern in _SIGNATURE_FILENAME_PATTERNS:
            if pattern.match(fn):
                # If it matches a signature pattern AND is small, it's a signature
                if size is not None and size < _SIGNATURE_MAX_SIZE:
                    return True

        return False

    def is_signature_attachment(self, filename, content_type, size):
        """Public API for checking signature attachments (used by dashboard)."""
        if not self.skip_signature_attachments:
            return False
        return self._is_signature_attachment(filename, content_type, size)

    def save_attachments(self, msg_data, transmittal_no):
        """Save all attachments from a parsed message to the transmittal folder.

        Returns a list of dicts: [{filename, path, size, content_type}]
        """
        attachments = msg_data.get('attachments', [])
        if not attachments:
            return []

        email_date = msg_data.get('date', datetime.now())
        if isinstance(email_date, str):
            from dateutil.parser import parse as parse_date
            email_date = parse_date(email_date)

        folder = self._create_folder(email_date, transmittal_no)
        saved = []

        for att in attachments:
            filename = _sanitize_filename(att.get('filename', 'unnamed'))
            data = att.get('data', b'')
            content_type = att.get('content_type', 'application/octet-stream')

            # Check size limit
            if len(data) > self.max_size_bytes:
                logger.warning("Skipping %s: size %d exceeds limit %d", filename, len(data), self.max_size_bytes)
                continue

            # Check allowed extensions
            if self.allowed_extensions:
                ext = os.path.splitext(filename)[1].lower()
                if ext and ext not in self.allowed_extensions:
                    logger.warning("Skipping %s: extension %s not allowed", filename, ext)
                    continue

            filepath = os.path.join(folder, filename)

            # Handle duplicate filenames
            if os.path.exists(filepath):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(filepath):
                    filepath = os.path.join(folder, f"{base}_{counter}{ext}")
                    counter += 1

            with open(filepath, 'wb') as f:
                f.write(data)

            saved.append({
                'filename': os.path.basename(filepath),
                'path': filepath,
                'size': len(data),
                'content_type': content_type,
            })
            logger.info("Saved attachment: %s (%d bytes)", filepath, len(data))

        # Write metadata
        self._write_metadata(folder, msg_data, transmittal_no, saved)

        return saved

    def _create_folder(self, email_date, transmittal_no):
        """Create folder: base_path/YYYY/MM-MonthName/TRN-YYYY-NNNN/"""
        year = str(email_date.year)
        month = f"{email_date.month:02d}-{MONTH_NAMES[email_date.month]}"
        folder = os.path.join(self.base_path, year, month, transmittal_no)
        os.makedirs(folder, exist_ok=True)
        return folder

    def _write_metadata(self, folder, msg_data, transmittal_no, saved_files):
        """Write _metadata.json with email context."""
        metadata = {
            'transmittal_no': transmittal_no,
            'email_date': str(msg_data.get('date', '')),
            'email_subject': msg_data.get('subject', ''),
            'sender': msg_data.get('sender', ''),
            'to_recipients': msg_data.get('to', []),
            'cc_recipients': msg_data.get('cc', []),
            'message_id': msg_data.get('message_id', ''),
            'attachments': [
                {'filename': f['filename'], 'size_bytes': f['size'], 'content_type': f['content_type']}
                for f in saved_files
            ],
        }
        path = os.path.join(folder, '_metadata.json')
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def save_single_attachment(self, filename, data, content_type, email_date, transmittal_no):
        """Save one attachment from a stored blob (used during approval from pending queue).

        Returns: {filename, path, size, content_type} or None if skipped.
        """
        if isinstance(email_date, str):
            from dateutil.parser import parse as parse_date
            email_date = parse_date(email_date)
        if email_date is None:
            email_date = datetime.now()

        filename = _sanitize_filename(filename)

        if len(data) > self.max_size_bytes:
            logger.warning("Skipping %s: size %d exceeds limit %d", filename, len(data), self.max_size_bytes)
            return None

        if self.allowed_extensions:
            ext = os.path.splitext(filename)[1].lower()
            if ext and ext not in self.allowed_extensions:
                logger.warning("Skipping %s: extension %s not allowed", filename, ext)
                return None

        folder = self._create_folder(email_date, transmittal_no)
        filepath = os.path.join(folder, filename)

        if os.path.exists(filepath):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filepath):
                filepath = os.path.join(folder, f"{base}_{counter}{ext}")
                counter += 1

        with open(filepath, 'wb') as f:
            f.write(data)

        logger.info("Saved attachment: %s (%d bytes)", filepath, len(data))
        return {
            'filename': os.path.basename(filepath),
            'path': filepath,
            'size': len(data),
            'content_type': content_type,
        }
