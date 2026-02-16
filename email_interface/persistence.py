import json
import logging
import os
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_messages (
    message_id TEXT PRIMARY KEY,
    transmittal_no TEXT NOT NULL,
    processed_at TIMESTAMP NOT NULL,
    sender TEXT,
    sender_name TEXT,
    to_recipients TEXT,
    cc_recipients TEXT,
    subject TEXT,
    attachment_count INTEGER,
    doc_type TEXT,
    discipline TEXT,
    department TEXT,
    response_required INTEGER DEFAULT 0,
    references_json TEXT
);

CREATE TABLE IF NOT EXISTS transmittal_sequence (
    year INTEGER PRIMARY KEY,
    last_sequence INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT UNIQUE,
    status TEXT DEFAULT 'pending_review',
    scanned_at TIMESTAMP,
    decided_at TIMESTAMP,
    sender TEXT,
    sender_name TEXT,
    to_recipients TEXT,
    cc_recipients TEXT,
    subject TEXT,
    email_date TIMESTAMP,
    body TEXT,
    doc_type TEXT,
    discipline TEXT,
    department TEXT,
    response_required INTEGER DEFAULT 0,
    references_json TEXT,
    attachment_count INTEGER DEFAULT 0,
    transmittal_no TEXT,
    attachment_folder TEXT
);

CREATE TABLE IF NOT EXISTS pending_attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pending_email_id INTEGER NOT NULL,
    filename TEXT,
    content_type TEXT,
    size INTEGER,
    data BLOB,
    FOREIGN KEY (pending_email_id) REFERENCES pending_emails(id)
);

CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    keywords TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    company TEXT,
    department TEXT,
    notes TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP
);
"""

# Columns added after initial schema - handled via migration
_MIGRATIONS = [
    ("processed_messages", "sender_name", "TEXT"),
    ("processed_messages", "doc_type", "TEXT"),
    ("processed_messages", "discipline", "TEXT"),
    ("processed_messages", "department", "TEXT"),
    ("processed_messages", "response_required", "INTEGER DEFAULT 0"),
    ("processed_messages", "references_json", "TEXT"),
    ("pending_emails", "classifier_method", "TEXT"),
    ("pending_emails", "confidence", "REAL"),
    ("departments", "description", "TEXT"),
    ("pending_emails", "body_html", "TEXT"),
]


class ProcessingTracker:
    """SQLite-backed tracker for processed emails and transmittal number sequences."""

    def __init__(self, db_path='data/tracking.db', prefix='TRN'):
        self.prefix = prefix
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._run_migrations()

    def _run_migrations(self):
        """Add columns that may be missing on older databases."""
        for table, column, col_type in _MIGRATIONS:
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                self._conn.commit()
                logger.info("Migration: added %s.%s", table, column)
            except sqlite3.OperationalError:
                pass  # Column already exists

    # --- Existing methods ---

    def is_processed(self, message_id):
        """Check if a message has already been processed."""
        cur = self._conn.execute(
            "SELECT 1 FROM processed_messages WHERE message_id = ?",
            (message_id,),
        )
        return cur.fetchone() is not None

    def mark_processed(self, message_id, transmittal_no, msg_data):
        """Record a processed message with full classification data."""
        refs = msg_data.get('references', [])
        self._conn.execute(
            """INSERT OR IGNORE INTO processed_messages
               (message_id, transmittal_no, processed_at, sender, sender_name,
                to_recipients, cc_recipients, subject, attachment_count,
                doc_type, discipline, department, response_required, references_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                message_id,
                transmittal_no,
                datetime.now().isoformat(),
                msg_data.get('sender', ''),
                msg_data.get('sender_name', ''),
                json.dumps(msg_data.get('to', [])),
                json.dumps(msg_data.get('cc', [])),
                msg_data.get('subject', ''),
                len(msg_data.get('attachments', [])),
                msg_data.get('doc_type', ''),
                msg_data.get('discipline', ''),
                msg_data.get('department', ''),
                1 if msg_data.get('response_required') else 0,
                json.dumps(refs) if isinstance(refs, list) else refs,
            ),
        )
        self._conn.commit()
        logger.info("Tracked: %s -> %s", transmittal_no, message_id)

    def get_next_transmittal_number(self, date=None):
        """Generate next transmittal number: TRN-YYYY-NNNN."""
        if date is None:
            date = datetime.now()
        year = date.year

        cur = self._conn.execute(
            "SELECT last_sequence FROM transmittal_sequence WHERE year = ?",
            (year,),
        )
        row = cur.fetchone()

        if row:
            seq = row[0] + 1
            self._conn.execute(
                "UPDATE transmittal_sequence SET last_sequence = ? WHERE year = ?",
                (seq, year),
            )
        else:
            seq = 1
            self._conn.execute(
                "INSERT INTO transmittal_sequence (year, last_sequence) VALUES (?, ?)",
                (year, seq),
            )

        self._conn.commit()
        return f"{self.prefix}-{year}-{seq:04d}"

    # --- Pending emails (Phase 1 - scan & queue) ---

    def is_pending(self, message_id):
        """Check if a message is already in the pending queue."""
        cur = self._conn.execute(
            "SELECT 1 FROM pending_emails WHERE message_id = ?",
            (message_id,),
        )
        return cur.fetchone() is not None

    def store_pending_email(self, msg_data, classification, status='pending_review'):
        """Store a scanned email as pending review. Returns the pending email id."""
        to_json = json.dumps(msg_data.get('to', []), default=str)
        cc_json = json.dumps(msg_data.get('cc', []), default=str)
        refs_json = json.dumps(classification.get('references', []))
        email_date = msg_data.get('date')
        if email_date and hasattr(email_date, 'isoformat'):
            email_date = email_date.isoformat()

        cur = self._conn.execute(
            """INSERT INTO pending_emails
               (message_id, status, scanned_at, sender, sender_name,
                to_recipients, cc_recipients, subject, email_date, body,
                doc_type, discipline, department, response_required,
                references_json, attachment_count, classifier_method, confidence,
                body_html)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg_data.get('message_id', ''),
                status,
                datetime.now().isoformat(),
                msg_data.get('sender', ''),
                msg_data.get('sender_name', ''),
                to_json,
                cc_json,
                msg_data.get('subject', ''),
                email_date,
                msg_data.get('body', ''),
                classification.get('doc_type', 'Others'),
                classification.get('discipline', 'General'),
                classification.get('department', ''),
                1 if classification.get('response_required') else 0,
                refs_json,
                len(msg_data.get('attachments', [])),
                classification.get('classifier_method', ''),
                classification.get('confidence'),
                msg_data.get('body_html', ''),
            ),
        )
        self._conn.commit()
        pending_id = cur.lastrowid
        logger.info("Stored pending email #%d: %s", pending_id, msg_data.get('subject', ''))
        return pending_id

    def store_pending_attachment(self, pending_email_id, filename, content_type, size, data):
        """Store an attachment blob for a pending email."""
        self._conn.execute(
            """INSERT INTO pending_attachments
               (pending_email_id, filename, content_type, size, data)
               VALUES (?, ?, ?, ?, ?)""",
            (pending_email_id, filename, content_type, size, data),
        )
        self._conn.commit()

    def get_pending_emails(self, status='pending_review', limit=100):
        """Get pending emails filtered by status."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            """SELECT id, message_id, status, scanned_at, sender, sender_name,
                      to_recipients, cc_recipients, subject, email_date,
                      doc_type, discipline, department, response_required,
                      references_json, attachment_count, transmittal_no,
                      classifier_method, confidence
               FROM pending_emails WHERE status = ?
               ORDER BY scanned_at DESC LIMIT ?""",
            (status, limit),
        )
        rows = [dict(row) for row in cur.fetchall()]
        self._conn.row_factory = None
        return rows

    def get_pending_email(self, pending_id):
        """Get a single pending email by id."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT * FROM pending_emails WHERE id = ?",
            (pending_id,),
        )
        row = cur.fetchone()
        self._conn.row_factory = None
        return dict(row) if row else None

    def get_pending_attachments(self, pending_email_id):
        """Get attachment blobs for a pending email."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT id, filename, content_type, size, data FROM pending_attachments WHERE pending_email_id = ?",
            (pending_email_id,),
        )
        rows = [dict(row) for row in cur.fetchall()]
        self._conn.row_factory = None
        return rows

    def get_pending_attachments_meta(self, pending_email_id):
        """Get attachment metadata (no BLOB) for listing in the dashboard."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT id, filename, content_type, size FROM pending_attachments WHERE pending_email_id = ?",
            (pending_email_id,),
        )
        rows = [dict(row) for row in cur.fetchall()]
        self._conn.row_factory = None
        return rows

    def get_single_pending_attachment(self, att_id, pending_email_id):
        """Get a single attachment with BLOB for download."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT id, filename, content_type, size, data FROM pending_attachments WHERE id = ? AND pending_email_id = ?",
            (att_id, pending_email_id),
        )
        row = cur.fetchone()
        self._conn.row_factory = None
        return dict(row) if row else None

    def approve_pending_email(self, pending_id, edits=None):
        """Approve a pending email: generate transmittal#, copy to processed_messages.

        Args:
            pending_id: The pending email id
            edits: Optional dict with keys doc_type, discipline, department to override

        Returns:
            dict with transmittal_no and the updated pending email row
        """
        pe = self.get_pending_email(pending_id)
        if not pe:
            raise ValueError(f"Pending email #{pending_id} not found")
        if pe['status'] != 'pending_review':
            raise ValueError(f"Pending email #{pending_id} is already {pe['status']}")

        # Apply user edits
        if edits:
            for key in ('doc_type', 'discipline', 'department'):
                if key in edits and edits[key] is not None:
                    pe[key] = edits[key]

        # Generate transmittal number
        email_date = None
        if pe.get('email_date'):
            try:
                from dateutil.parser import parse as parse_date
                email_date = parse_date(pe['email_date'])
            except Exception:
                pass
        transmittal_no = self.get_next_transmittal_number(email_date)

        # Copy to processed_messages (with full classification)
        self._conn.execute(
            """INSERT OR IGNORE INTO processed_messages
               (message_id, transmittal_no, processed_at, sender, sender_name,
                to_recipients, cc_recipients, subject, attachment_count,
                doc_type, discipline, department, response_required, references_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pe['message_id'],
                transmittal_no,
                datetime.now().isoformat(),
                pe['sender'],
                pe.get('sender_name', ''),
                pe.get('to_recipients', '[]'),
                pe.get('cc_recipients', '[]'),
                pe['subject'],
                pe['attachment_count'],
                pe.get('doc_type', ''),
                pe.get('discipline', ''),
                pe.get('department', ''),
                pe.get('response_required', 0),
                pe.get('references_json', '[]'),
            ),
        )

        # Update pending status
        self._conn.execute(
            """UPDATE pending_emails
               SET status = 'approved', decided_at = ?, transmittal_no = ?,
                   doc_type = ?, discipline = ?, department = ?
               WHERE id = ?""",
            (
                datetime.now().isoformat(),
                transmittal_no,
                pe['doc_type'],
                pe['discipline'],
                pe.get('department', ''),
                pending_id,
            ),
        )
        self._conn.commit()

        pe['transmittal_no'] = transmittal_no
        pe['status'] = 'approved'
        logger.info("Approved pending #%d -> %s", pending_id, transmittal_no)
        return {'transmittal_no': transmittal_no, 'pending_email': pe}

    def reject_pending_email(self, pending_id):
        """Reject a pending email."""
        pe = self.get_pending_email(pending_id)
        if not pe:
            raise ValueError(f"Pending email #{pending_id} not found")
        if pe['status'] != 'pending_review':
            raise ValueError(f"Pending email #{pending_id} is already {pe['status']}")

        self._conn.execute(
            "UPDATE pending_emails SET status = 'rejected', decided_at = ? WHERE id = ?",
            (datetime.now().isoformat(), pending_id),
        )
        self._conn.commit()
        logger.info("Rejected pending #%d: %s", pending_id, pe.get('subject', ''))

    def get_pending_stats(self):
        """Get counts of pending emails by status."""
        cur = self._conn.execute(
            "SELECT status, COUNT(*) FROM pending_emails GROUP BY status"
        )
        stats = {row[0]: row[1] for row in cur.fetchall()}
        return {
            'pending_review': stats.get('pending_review', 0),
            'approved': stats.get('approved', 0),
            'rejected': stats.get('rejected', 0),
            'out_of_scope': stats.get('out_of_scope', 0),
        }

    def move_to_review(self, pending_id):
        """Move an out-of-scope email back to pending_review."""
        pe = self.get_pending_email(pending_id)
        if not pe:
            raise ValueError(f"Pending email #{pending_id} not found")
        if pe['status'] != 'out_of_scope':
            raise ValueError(f"Email #{pending_id} is {pe['status']}, not out_of_scope")
        self._conn.execute(
            "UPDATE pending_emails SET status = 'pending_review', decided_at = NULL WHERE id = ?",
            (pending_id,)
        )
        self._conn.commit()
        logger.info("Moved to review: pending #%d", pending_id)

    def reopen_rejected(self, pending_id):
        """Reopen a rejected email back to pending_review."""
        pe = self.get_pending_email(pending_id)
        if not pe:
            raise ValueError(f"Pending email #{pending_id} not found")
        if pe['status'] != 'rejected':
            raise ValueError(f"Email #{pending_id} is {pe['status']}, not rejected")
        self._conn.execute(
            "UPDATE pending_emails SET status = 'pending_review', decided_at = NULL WHERE id = ?",
            (pending_id,)
        )
        self._conn.commit()
        logger.info("Reopened rejected: pending #%d", pending_id)

    # --- Departments ---

    def get_departments(self, active_only=False):
        """Get all departments."""
        self._conn.row_factory = sqlite3.Row
        query = "SELECT * FROM departments"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY name"
        cur = self._conn.execute(query)
        rows = [dict(row) for row in cur.fetchall()]
        self._conn.row_factory = None
        return rows

    def add_department(self, name, keywords=None, description=None):
        """Add a new department. keywords is a list of strings."""
        kw_json = json.dumps(keywords or [])
        now = datetime.now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO departments (name, keywords, description, is_active, created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?)",
            (name, kw_json, description or '', now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_department(self, dept_id, name=None, keywords=None, is_active=None, description=None):
        """Update a department."""
        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if keywords is not None:
            updates.append("keywords = ?")
            params.append(json.dumps(keywords))
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(1 if is_active else 0)
        if not updates:
            return
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(dept_id)
        self._conn.execute(
            f"UPDATE departments SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        self._conn.commit()

    def delete_department(self, dept_id):
        """Delete a department."""
        self._conn.execute("DELETE FROM departments WHERE id = ?", (dept_id,))
        self._conn.commit()

    def seed_default_departments(self, discipline_keywords):
        """One-time seed from discipline_keywords config if departments table is empty."""
        cur = self._conn.execute("SELECT COUNT(*) FROM departments")
        if cur.fetchone()[0] > 0:
            return
        now = datetime.now().isoformat()
        for name, keywords in discipline_keywords.items():
            self._conn.execute(
                "INSERT OR IGNORE INTO departments (name, keywords, is_active, created_at, updated_at) VALUES (?, ?, 1, ?, ?)",
                (name, json.dumps(keywords), now, now),
            )
        self._conn.commit()
        logger.info("Seeded %d default departments from discipline_keywords", len(discipline_keywords))

    # --- Contacts (expected senders) ---

    def get_contacts(self, active_only=False):
        """Get all contacts."""
        self._conn.row_factory = sqlite3.Row
        query = "SELECT * FROM contacts"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY company, name"
        cur = self._conn.execute(query)
        rows = [dict(row) for row in cur.fetchall()]
        self._conn.row_factory = None
        return rows

    def add_contact(self, name, email, company='', department='', notes=''):
        """Add a new contact. Returns the contact id."""
        now = datetime.now().isoformat()
        cur = self._conn.execute(
            """INSERT INTO contacts (name, email, company, department, notes, is_active, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
            (name, email.lower().strip(), company, department, notes, now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_contact(self, contact_id, name=None, email=None, company=None,
                       department=None, notes=None, is_active=None):
        """Update a contact."""
        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if email is not None:
            updates.append("email = ?")
            params.append(email.lower().strip())
        if company is not None:
            updates.append("company = ?")
            params.append(company)
        if department is not None:
            updates.append("department = ?")
            params.append(department)
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(1 if is_active else 0)
        if not updates:
            return
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(contact_id)
        self._conn.execute(
            f"UPDATE contacts SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        self._conn.commit()

    def delete_contact(self, contact_id):
        """Delete a contact."""
        self._conn.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
        self._conn.commit()

    def find_contact_by_email(self, email):
        """Look up a contact by email address. Returns dict or None."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT * FROM contacts WHERE email = ? AND is_active = 1",
            (email.lower().strip(),),
        )
        row = cur.fetchone()
        self._conn.row_factory = None
        return dict(row) if row else None

    def find_contacts_by_emails(self, emails):
        """Look up multiple contacts by email. Returns {email: contact_dict}."""
        result = {}
        for em in emails:
            contact = self.find_contact_by_email(em)
            if contact:
                result[em.lower().strip()] = contact
        return result

    # --- Settings (key/value store) ---

    def get_setting(self, key, default=None):
        """Get a setting value by key. Returns default if not found."""
        cur = self._conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return row[0] if row else default

    def set_setting(self, key, value):
        """Set a setting value (upsert)."""
        self._conn.execute(
            """INSERT INTO settings (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
            (key, value, datetime.now().isoformat()),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()
