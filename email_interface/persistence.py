import json
import logging
import os
import re
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

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    normalized_subject TEXT NOT NULL,
    transmittal_no TEXT,
    email_count INTEGER DEFAULT 1,
    first_message_id TEXT,
    last_message_id TEXT,
    first_date TIMESTAMP,
    last_date TIMESTAMP,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_number TEXT NOT NULL,
    variant TEXT DEFAULT '',
    client_name TEXT DEFAULT '',
    year INTEGER,
    folder_path TEXT UNIQUE NOT NULL,
    template_type TEXT DEFAULT 'unknown',
    status TEXT DEFAULT 'active',
    file_count INTEGER DEFAULT 0,
    subfolder_json TEXT DEFAULT '{}',
    notes TEXT DEFAULT '',
    scanned_at TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_sequences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    doc_type_code TEXT NOT NULL,
    last_sequence INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    UNIQUE(project_id, doc_type_code)
);

CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_emails(status);
CREATE INDEX IF NOT EXISTS idx_pending_scanned_at ON pending_emails(scanned_at);
CREATE INDEX IF NOT EXISTS idx_pending_conversation ON pending_emails(conversation_id);
CREATE INDEX IF NOT EXISTS idx_processed_conversation ON processed_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conv_norm_subject ON conversations(normalized_subject);
CREATE TABLE IF NOT EXISTS sent_emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT,
    transmittal_no TEXT,
    recipient TEXT NOT NULL,
    email_type TEXT NOT NULL,
    subject TEXT,
    body TEXT,
    sent_at TIMESTAMP,
    status TEXT DEFAULT 'sent',
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_projects_number ON projects(project_number);
CREATE INDEX IF NOT EXISTS idx_projects_year ON projects(year);
CREATE INDEX IF NOT EXISTS idx_sent_emails_transmittal ON sent_emails(transmittal_no);
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
    ("pending_emails", "in_reply_to", "TEXT"),
    ("pending_emails", "email_references", "TEXT"),
    ("pending_emails", "conversation_id", "INTEGER"),
    ("pending_emails", "conversation_position", "INTEGER DEFAULT 1"),
    ("processed_messages", "conversation_id", "INTEGER"),
    ("pending_emails", "ai_summary", "TEXT"),
    ("pending_emails", "ai_priority", "TEXT DEFAULT 'medium'"),
    ("pending_emails", "user_corrections", "TEXT"),
    ("pending_emails", "auto_approved", "INTEGER DEFAULT 0"),
    ("contacts", "is_team_member", "INTEGER DEFAULT 0"),
    ("contacts", "role", "TEXT"),
    ("pending_emails", "assigned_to", "TEXT"),
    ("processed_messages", "assigned_to", "TEXT"),
    ("pending_emails", "acknowledgment_required", "INTEGER DEFAULT 0"),
    ("processed_messages", "acknowledgment_required", "INTEGER DEFAULT 0"),
    ("pending_emails", "acknowledgment_sent_at", "TIMESTAMP"),
    ("processed_messages", "acknowledgment_sent_at", "TIMESTAMP"),
    ("processed_messages", "reminder_sent_at", "TIMESTAMP"),
    ("processed_messages", "reminder_count", "INTEGER DEFAULT 0"),
    ("processed_messages", "response_received", "INTEGER DEFAULT 0"),
    ("processed_messages", "response_due_date", "TIMESTAMP"),
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
                doc_type, discipline, department, response_required, references_json,
                conversation_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                msg_data.get('conversation_id'),
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
                body_html, in_reply_to, email_references,
                conversation_id, conversation_position,
                ai_summary, ai_priority, acknowledgment_required)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                msg_data.get('in_reply_to', ''),
                msg_data.get('email_references', ''),
                msg_data.get('conversation_id'),
                msg_data.get('conversation_position', 1),
                classification.get('summary', ''),
                classification.get('priority', 'medium'),
                1 if classification.get('acknowledgment_required') else 0,
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
                      acknowledgment_required,
                      references_json, attachment_count, transmittal_no,
                      classifier_method, confidence,
                      conversation_id, conversation_position,
                      ai_summary, ai_priority, assigned_to
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
            for key in ('doc_type', 'discipline', 'department', 'assigned_to'):
                if key in edits and edits[key] is not None:
                    pe[key] = edits[key]
            if 'acknowledgment_required' in edits:
                pe['acknowledgment_required'] = 1 if edits['acknowledgment_required'] else 0

        # Generate transmittal number
        email_date = None
        if pe.get('email_date'):
            try:
                from dateutil.parser import parse as parse_date
                email_date = parse_date(pe['email_date'])
            except Exception as exc:
                logger.debug("Could not parse email_date '%s': %s", pe['email_date'], exc)
        transmittal_no = self.get_next_transmittal_number(email_date)

        # Copy to processed_messages (with full classification)
        # Set response_due_date if response is required
        response_due = None
        if pe.get('response_required'):
            from datetime import timedelta
            response_due = (datetime.now() + timedelta(days=3)).isoformat()

        self._conn.execute(
            """INSERT OR IGNORE INTO processed_messages
               (message_id, transmittal_no, processed_at, sender, sender_name,
                to_recipients, cc_recipients, subject, attachment_count,
                doc_type, discipline, department, response_required, references_json,
                conversation_id, assigned_to, acknowledgment_required, response_due_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                pe.get('conversation_id'),
                pe.get('assigned_to', ''),
                pe.get('acknowledgment_required', 0),
                response_due,
            ),
        )

        # Update pending status
        now = datetime.now().isoformat()
        self._conn.execute(
            """UPDATE pending_emails
               SET status = 'approved', decided_at = ?, transmittal_no = ?,
                   doc_type = ?, discipline = ?, department = ?,
                   assigned_to = ?
               WHERE id = ?""",
            (
                now,
                transmittal_no,
                pe['doc_type'],
                pe['discipline'],
                pe.get('department', ''),
                pe.get('assigned_to', ''),
                pending_id,
            ),
        )

        # Propagate transmittal to conversation and auto-approve siblings
        auto_approved = []
        conv_id = pe.get('conversation_id')
        if conv_id:
            self._conn.execute(
                "UPDATE conversations SET transmittal_no = ? WHERE id = ? AND transmittal_no IS NULL",
                (transmittal_no, conv_id),
            )
            # Find sibling pending emails in same conversation
            cur_siblings = self._conn.execute(
                """SELECT id FROM pending_emails
                   WHERE conversation_id = ? AND id != ? AND status = 'pending_review'""",
                (conv_id, pending_id),
            )
            sibling_ids = [row[0] for row in cur_siblings.fetchall()]
            for sib_id in sibling_ids:
                sib = self.get_pending_email(sib_id)
                if not sib:
                    continue
                self._conn.execute(
                    """UPDATE pending_emails
                       SET status = 'approved', decided_at = ?, transmittal_no = ?
                       WHERE id = ?""",
                    (now, transmittal_no, sib_id),
                )
                # Copy sibling to processed_messages
                self._conn.execute(
                    """INSERT OR IGNORE INTO processed_messages
                       (message_id, transmittal_no, processed_at, sender, sender_name,
                        to_recipients, cc_recipients, subject, attachment_count,
                        doc_type, discipline, department, response_required,
                        references_json, conversation_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sib['message_id'], transmittal_no, now,
                        sib['sender'], sib.get('sender_name', ''),
                        sib.get('to_recipients', '[]'), sib.get('cc_recipients', '[]'),
                        sib['subject'], sib['attachment_count'],
                        sib.get('doc_type', ''), sib.get('discipline', ''),
                        sib.get('department', ''), sib.get('response_required', 0),
                        sib.get('references_json', '[]'), conv_id,
                    ),
                )
                auto_approved.append(sib_id)
                logger.info("Auto-approved sibling #%d -> %s", sib_id, transmittal_no)

        self._conn.commit()

        pe['transmittal_no'] = transmittal_no
        pe['status'] = 'approved'
        logger.info("Approved pending #%d -> %s", pending_id, transmittal_no)
        return {'transmittal_no': transmittal_no, 'pending_email': pe, 'auto_approved': auto_approved}

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
                       department=None, notes=None, is_active=None,
                       is_team_member=None, role=None):
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
        if is_team_member is not None:
            updates.append("is_team_member = ?")
            params.append(1 if is_team_member else 0)
        if role is not None:
            updates.append("role = ?")
            params.append(role)
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

    def get_team_members(self):
        """Get all active contacts marked as team members."""
        self._conn.row_factory = sqlite3.Row
        try:
            cur = self._conn.execute(
                "SELECT * FROM contacts WHERE is_team_member = 1 AND is_active = 1 ORDER BY name"
            )
            rows = [dict(row) for row in cur.fetchall()]
        except sqlite3.OperationalError:
            rows = []
        self._conn.row_factory = None
        return rows

    def find_contacts_by_emails(self, emails):
        """Look up multiple contacts by email. Returns {email: contact_dict}."""
        result = {}
        for em in emails:
            contact = self.find_contact_by_email(em)
            if contact:
                result[em.lower().strip()] = contact
        return result

    # --- Accuracy & Correction Stats ---

    def get_accuracy_stats(self):
        """Per-classifier accuracy based on user corrections.

        Returns dict with per_classifier list and common_corrections patterns.
        """
        per_classifier = []
        try:
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.execute(
                """SELECT classifier_method,
                          COUNT(*) as total,
                          SUM(CASE WHEN user_corrections IS NOT NULL THEN 1 ELSE 0 END) as corrected,
                          AVG(confidence) as avg_confidence
                   FROM pending_emails
                   WHERE status = 'approved' AND auto_approved = 0
                         AND classifier_method IS NOT NULL AND classifier_method != ''
                   GROUP BY classifier_method
                   ORDER BY total DESC"""
            )
            for row in cur.fetchall():
                total = row['total']
                corrected = row['corrected']
                accuracy = round((total - corrected) / total * 100, 1) if total > 0 else 0
                per_classifier.append({
                    'method': row['classifier_method'],
                    'total': total,
                    'corrected': corrected,
                    'accuracy_pct': accuracy,
                    'avg_confidence': round(row['avg_confidence'], 2) if row['avg_confidence'] else 0,
                })
            self._conn.row_factory = None
        except sqlite3.OperationalError:
            pass

        # Common corrections patterns
        common_corrections = {}
        try:
            cur = self._conn.execute(
                """SELECT user_corrections FROM pending_emails
                   WHERE user_corrections IS NOT NULL AND status = 'approved'"""
            )
            for row in cur.fetchall():
                try:
                    corr = json.loads(row[0])
                    for field, (old_val, new_val) in corr.items():
                        if field not in common_corrections:
                            common_corrections[field] = {}
                        key = f"{old_val or '(empty)'} -> {new_val}"
                        common_corrections[field][key] = common_corrections[field].get(key, 0) + 1
                except (json.JSONDecodeError, ValueError):
                    pass
        except sqlite3.OperationalError:
            pass

        # Sort corrections by frequency
        for field in common_corrections:
            items = sorted(common_corrections[field].items(), key=lambda x: x[1], reverse=True)
            common_corrections[field] = [{'pattern': k, 'count': v} for k, v in items[:10]]

        return {
            'per_classifier': per_classifier,
            'common_corrections': common_corrections,
        }

    # --- Contact Learning Queries ---

    def count_approved_from_sender(self, email):
        """Count how many approved emails are from this sender."""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM pending_emails WHERE LOWER(sender) = ? AND status = 'approved'",
            (email.lower().strip(),),
        )
        return cur.fetchone()[0]

    def get_most_common_department_for_sender(self, email):
        """Get the most frequently assigned department for a sender."""
        cur = self._conn.execute(
            """SELECT department, COUNT(*) as cnt FROM pending_emails
               WHERE LOWER(sender) = ? AND status = 'approved'
                     AND department IS NOT NULL AND department != ''
               GROUP BY department ORDER BY cnt DESC LIMIT 1""",
            (email.lower().strip(),),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def get_suggested_contacts(self, min_emails=2):
        """Find senders NOT in contacts but with min_emails+ approved emails.

        Returns list of dicts: sender, sender_name, email_count, department, company.
        """
        results = []
        try:
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.execute(
                """SELECT sender, sender_name, COUNT(*) as email_count
                   FROM pending_emails
                   WHERE status = 'approved'
                     AND sender IS NOT NULL AND sender != ''
                     AND LOWER(sender) NOT IN (
                         SELECT LOWER(email) FROM contacts WHERE is_active = 1
                     )
                   GROUP BY LOWER(sender)
                   HAVING COUNT(*) >= ?
                   ORDER BY email_count DESC LIMIT 20""",
                (min_emails,),
            )
            for row in cur.fetchall():
                sender = row['sender']
                dept = self.get_most_common_department_for_sender(sender) or ''
                # Extract company from domain
                company = ''
                if sender and '@' in sender:
                    domain = sender.split('@')[1].lower()
                    parts = domain.split('.')
                    if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'ac'):
                        name = parts[-3]
                    elif len(parts) >= 2:
                        name = parts[-2]
                    else:
                        name = parts[0]
                    generic = {'gmail', 'yahoo', 'hotmail', 'outlook', 'live', 'aol', 'icloud', 'mail', 'protonmail'}
                    if name not in generic:
                        company = name.capitalize()
                results.append({
                    'sender': sender,
                    'sender_name': row['sender_name'] or '',
                    'email_count': row['email_count'],
                    'department': dept,
                    'company': company,
                })
            self._conn.row_factory = None
        except sqlite3.OperationalError:
            pass
        return results

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

    # --- Sent Emails / Acknowledgment / Reminder Tracking ---

    def log_sent_email(self, message_id, transmittal_no, recipient, email_type,
                       subject, body, status='sent', error=None):
        """Log a sent email (acknowledgment or reminder)."""
        self._conn.execute(
            """INSERT INTO sent_emails
               (message_id, transmittal_no, recipient, email_type, subject, body,
                sent_at, status, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (message_id, transmittal_no, recipient, email_type, subject, body,
             datetime.now().isoformat(), status, error),
        )
        self._conn.commit()

    def get_sent_emails(self, transmittal_no=None, email_type=None, limit=50):
        """Get sent email log with optional filters."""
        self._conn.row_factory = sqlite3.Row
        query = "SELECT * FROM sent_emails WHERE 1=1"
        params = []
        if transmittal_no:
            query += " AND transmittal_no = ?"
            params.append(transmittal_no)
        if email_type:
            query += " AND email_type = ?"
            params.append(email_type)
        query += " ORDER BY sent_at DESC LIMIT ?"
        params.append(limit)
        cur = self._conn.execute(query, params)
        rows = [dict(row) for row in cur.fetchall()]
        self._conn.row_factory = None
        return rows

    def get_overdue_responses(self, days=3):
        """Get processed messages requiring a response that are overdue."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            """SELECT * FROM processed_messages
               WHERE response_required = 1
                 AND response_received = 0
                 AND response_due_date IS NOT NULL
                 AND response_due_date < ?
               ORDER BY response_due_date ASC""",
            (datetime.now().isoformat(),),
        )
        rows = [dict(row) for row in cur.fetchall()]
        self._conn.row_factory = None
        return rows

    def mark_response_received(self, message_id):
        """Mark a processed message as having received a response."""
        self._conn.execute(
            "UPDATE processed_messages SET response_received = 1 WHERE message_id = ?",
            (message_id,),
        )
        self._conn.commit()

    def update_reminder_sent(self, message_id):
        """Increment reminder_count and set reminder_sent_at."""
        self._conn.execute(
            """UPDATE processed_messages
               SET reminder_count = COALESCE(reminder_count, 0) + 1,
                   reminder_sent_at = ?
               WHERE message_id = ?""",
            (datetime.now().isoformat(), message_id),
        )
        self._conn.commit()

    def mark_acknowledgment_sent(self, message_id, pending_id=None):
        """Set acknowledgment_sent_at on both processed and pending records."""
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE processed_messages SET acknowledgment_sent_at = ? WHERE message_id = ?",
            (now, message_id),
        )
        if pending_id:
            self._conn.execute(
                "UPDATE pending_emails SET acknowledgment_sent_at = ? WHERE id = ?",
                (now, pending_id),
            )
        self._conn.commit()

    def get_overdue_stats(self):
        """Get overdue response counts for dashboard KPI."""
        try:
            total = self._conn.execute(
                """SELECT COUNT(*) FROM processed_messages
                   WHERE response_required = 1
                     AND response_received = 0
                     AND response_due_date IS NOT NULL
                     AND response_due_date < ?""",
                (datetime.now().isoformat(),),
            ).fetchone()[0]
        except sqlite3.OperationalError:
            total = 0
        return {'overdue_responses': total}

    # --- Conversation Threading ---

    def _normalize_subject(self, subject):
        """Strip RE:/FW:/FWD: prefixes and normalize whitespace."""
        if not subject:
            return ''
        cleaned = re.sub(r'^(\s*(?:re|fw|fwd)\s*:\s*)+', '', subject, flags=re.IGNORECASE)
        return ' '.join(cleaned.split()).strip()

    def find_or_create_conversation(self, msg_data):
        """Find an existing conversation or create a new one.

        Returns: (conversation_id, conversation_position, transmittal_no_or_None)
        """
        in_reply_to = msg_data.get('in_reply_to', '').strip()
        email_refs = msg_data.get('email_references', '').strip()
        subject = msg_data.get('subject', '')
        message_id = msg_data.get('message_id', '')
        email_date = msg_data.get('date')
        if email_date and hasattr(email_date, 'isoformat'):
            email_date = email_date.isoformat()

        # Parse reference message IDs from headers
        ref_ids = []
        if in_reply_to:
            ref_ids.append(in_reply_to)
        if email_refs:
            ref_ids.extend(email_refs.split())
        # Deduplicate while preserving order
        seen = set()
        unique_refs = []
        for rid in ref_ids:
            rid = rid.strip()
            if rid and rid not in seen:
                seen.add(rid)
                unique_refs.append(rid)

        conv_id = None

        # Strategy A: Header match — look up referenced message IDs
        if unique_refs:
            placeholders = ','.join(['?'] * len(unique_refs))
            # Check pending_emails
            cur = self._conn.execute(
                f"SELECT conversation_id FROM pending_emails WHERE message_id IN ({placeholders}) AND conversation_id IS NOT NULL LIMIT 1",
                unique_refs,
            )
            row = cur.fetchone()
            if row:
                conv_id = row[0]
            else:
                # Check processed_messages
                cur = self._conn.execute(
                    f"SELECT conversation_id FROM processed_messages WHERE message_id IN ({placeholders}) AND conversation_id IS NOT NULL LIMIT 1",
                    unique_refs,
                )
                row = cur.fetchone()
                if row:
                    conv_id = row[0]

        # Strategy B: Subject match
        if not conv_id:
            norm_subj = self._normalize_subject(subject)
            if norm_subj:
                cur = self._conn.execute(
                    "SELECT id FROM conversations WHERE normalized_subject = ? LIMIT 1",
                    (norm_subj,),
                )
                row = cur.fetchone()
                if row:
                    conv_id = row[0]

        # Strategy C: Create new conversation
        if not conv_id:
            norm_subj = self._normalize_subject(subject)
            now = datetime.now().isoformat()
            cur = self._conn.execute(
                """INSERT INTO conversations
                   (normalized_subject, email_count, first_message_id, last_message_id,
                    first_date, last_date, created_at)
                   VALUES (?, 1, ?, ?, ?, ?, ?)""",
                (norm_subj, message_id, message_id, email_date, email_date, now),
            )
            self._conn.commit()
            conv_id = cur.lastrowid
            return (conv_id, 1, None)

        # Update existing conversation counters
        self._conn.execute(
            """UPDATE conversations
               SET email_count = email_count + 1,
                   last_message_id = ?,
                   last_date = ?
               WHERE id = ?""",
            (message_id, email_date, conv_id),
        )
        self._conn.commit()

        # Get conversation position and transmittal
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT email_count, transmittal_no FROM conversations WHERE id = ?",
            (conv_id,),
        )
        row = cur.fetchone()
        self._conn.row_factory = None
        position = row['email_count'] if row else 1
        transmittal = row['transmittal_no'] if row else None

        return (conv_id, position, transmittal)

    def get_conversation_classification(self, conversation_id):
        """Get classification from the first classified email in a conversation.

        Returns: dict with classification fields, or None if no classified email found.
        """
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            """SELECT doc_type, discipline, department, response_required,
                      references_json, confidence, ai_summary, ai_priority
               FROM pending_emails
               WHERE conversation_id = ? AND status IN ('pending_review', 'approved')
                     AND doc_type IS NOT NULL AND doc_type != ''
               ORDER BY conversation_position ASC LIMIT 1""",
            (conversation_id,),
        )
        row = cur.fetchone()
        self._conn.row_factory = None
        if not row:
            return None
        return {
            'doc_type': row['doc_type'],
            'discipline': row['discipline'],
            'department': row['department'],
            'response_required': row['response_required'],
            'references': json.loads(row['references_json'] or '[]'),
            'confidence': row['confidence'],
            'summary': row['ai_summary'] or '',
            'priority': row['ai_priority'] or 'medium',
        }

    def get_conversation(self, conversation_id):
        """Return a conversation row as dict."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        row = cur.fetchone()
        self._conn.row_factory = None
        return dict(row) if row else None

    def get_conversation_emails(self, conversation_id):
        """Return all emails in a conversation (pending + processed) ordered by date."""
        emails = []
        self._conn.row_factory = sqlite3.Row

        # From pending_emails
        cur = self._conn.execute(
            """SELECT id, message_id, status, sender, sender_name, subject,
                      email_date, transmittal_no, conversation_position,
                      'pending' as source
               FROM pending_emails
               WHERE conversation_id = ?
               ORDER BY email_date""",
            (conversation_id,),
        )
        for row in cur.fetchall():
            emails.append(dict(row))

        # From processed_messages (only those not already in pending)
        cur = self._conn.execute(
            """SELECT message_id, sender, sender_name, subject,
                      processed_at as email_date, transmittal_no,
                      'processed' as source
               FROM processed_messages
               WHERE conversation_id = ?
                 AND message_id NOT IN (
                     SELECT message_id FROM pending_emails WHERE conversation_id = ?
                 )
               ORDER BY processed_at""",
            (conversation_id, conversation_id),
        )
        for row in cur.fetchall():
            emails.append(dict(row))

        self._conn.row_factory = None
        # Sort all by date
        emails.sort(key=lambda e: e.get('email_date') or '')
        return emails

    def get_thread_count(self, conversation_id):
        """Return the email_count for a conversation."""
        if not conversation_id:
            return 0
        cur = self._conn.execute(
            "SELECT email_count FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        row = cur.fetchone()
        return row[0] if row else 0

    def backfill_conversations(self):
        """One-time backfill: group existing emails without conversation_id into conversations."""
        logger.info("Starting conversation backfill...")
        count = 0

        # Process pending_emails without conversation_id
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            """SELECT id, message_id, subject, email_date, in_reply_to, email_references,
                      transmittal_no, status
               FROM pending_emails
               WHERE conversation_id IS NULL
               ORDER BY email_date"""
        )
        pending_rows = [dict(row) for row in cur.fetchall()]

        # Process processed_messages without conversation_id
        cur = self._conn.execute(
            """SELECT message_id, subject, processed_at as email_date, transmittal_no
               FROM processed_messages
               WHERE conversation_id IS NULL
               ORDER BY processed_at"""
        )
        processed_rows = [dict(row) for row in cur.fetchall()]
        self._conn.row_factory = None

        # Backfill pending emails
        for row in pending_rows:
            msg_data = {
                'message_id': row['message_id'],
                'subject': row['subject'],
                'date': row['email_date'],
                'in_reply_to': row.get('in_reply_to', '') or '',
                'email_references': row.get('email_references', '') or '',
            }
            conv_id, conv_pos, _ = self.find_or_create_conversation(msg_data)
            self._conn.execute(
                "UPDATE pending_emails SET conversation_id = ?, conversation_position = ? WHERE id = ?",
                (conv_id, conv_pos, row['id']),
            )
            # If this email has a transmittal, set it on the conversation
            if row.get('transmittal_no') and row.get('status') == 'approved':
                self._conn.execute(
                    "UPDATE conversations SET transmittal_no = ? WHERE id = ? AND transmittal_no IS NULL",
                    (row['transmittal_no'], conv_id),
                )
            count += 1

        # Backfill processed messages
        for row in processed_rows:
            msg_data = {
                'message_id': row['message_id'],
                'subject': row['subject'],
                'date': row['email_date'],
                'in_reply_to': '',
                'email_references': '',
            }
            conv_id, conv_pos, _ = self.find_or_create_conversation(msg_data)
            self._conn.execute(
                "UPDATE processed_messages SET conversation_id = ? WHERE message_id = ?",
                (conv_id, row['message_id']),
            )
            # Set transmittal on conversation
            if row.get('transmittal_no'):
                self._conn.execute(
                    "UPDATE conversations SET transmittal_no = ? WHERE id = ? AND transmittal_no IS NULL",
                    (row['transmittal_no'], conv_id),
                )
            count += 1

        self._conn.commit()
        logger.info("Backfill complete: %d emails assigned to conversations", count)
        return count

    # --- Projects ---

    def get_projects(self, year=None, status=None, search=None):
        """Get projects with optional filters. ORDER BY year DESC, project_number DESC."""
        clauses = []
        params = []
        if year is not None:
            clauses.append("year = ?")
            params.append(year)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if search:
            clauses.append("(project_number LIKE ? OR client_name LIKE ? OR notes LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like, like])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            f"SELECT * FROM projects{where} ORDER BY year DESC, project_number DESC",
            params,
        )
        rows = [dict(row) for row in cur.fetchall()]
        self._conn.row_factory = None
        return rows

    def get_project(self, project_id):
        """Get a single project by id."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cur.fetchone()
        self._conn.row_factory = None
        return dict(row) if row else None

    def add_project(self, project_number, variant='', client_name='', year=None,
                    folder_path='', template_type='unknown'):
        """Insert a new project."""
        now = datetime.now().isoformat()
        cur = self._conn.execute(
            """INSERT INTO projects
               (project_number, variant, client_name, year, folder_path,
                template_type, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
            (project_number, variant, client_name, year, folder_path,
             template_type, now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_project(self, project_id, **kwargs):
        """Update project fields dynamically."""
        allowed = {'project_number', 'variant', 'client_name', 'year', 'folder_path',
                   'template_type', 'status', 'file_count', 'subfolder_json',
                   'notes', 'scanned_at'}
        updates = []
        params = []
        for key, val in kwargs.items():
            if key in allowed:
                updates.append(f"{key} = ?")
                params.append(val)
        if not updates:
            return
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(project_id)
        self._conn.execute(
            f"UPDATE projects SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        self._conn.commit()

    def delete_project(self, project_id):
        """Delete a project and its document sequences."""
        self._conn.execute("DELETE FROM document_sequences WHERE project_id = ?", (project_id,))
        self._conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self._conn.commit()

    def get_project_by_folder(self, folder_path):
        """Look up a project by unique folder_path."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT * FROM projects WHERE folder_path = ?", (folder_path,)
        )
        row = cur.fetchone()
        self._conn.row_factory = None
        return dict(row) if row else None

    def get_project_stats(self):
        """Aggregate stats: total, by_year counts, by_status counts, with_coc count."""
        stats = {'total': 0, 'by_year': {}, 'by_status': {}, 'with_coc': 0, 'total_files': 0}
        cur = self._conn.execute("SELECT COUNT(*) FROM projects")
        stats['total'] = cur.fetchone()[0]

        cur = self._conn.execute(
            "SELECT year, COUNT(*) as cnt FROM projects GROUP BY year ORDER BY year DESC"
        )
        for row in cur.fetchall():
            stats['by_year'][row[0] or 'Unknown'] = row[1]

        cur = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM projects GROUP BY status"
        )
        for row in cur.fetchall():
            stats['by_status'][row[0] or 'unknown'] = row[1]

        cur = self._conn.execute(
            "SELECT COUNT(*) FROM projects WHERE subfolder_json LIKE '%Cert of Completion%' AND subfolder_json NOT LIKE '%\"count\": 0%'"
        )
        stats['with_coc'] = cur.fetchone()[0]

        cur = self._conn.execute("SELECT COALESCE(SUM(file_count), 0) FROM projects")
        stats['total_files'] = cur.fetchone()[0]

        return stats

    # --- Document Sequences ---

    def get_next_doc_number(self, project_id, doc_type_code):
        """UPSERT sequence and return formatted doc number like '150229-LTR-003'."""
        project = self.get_project(project_id)
        if not project:
            return None
        now = datetime.now().isoformat()
        # UPSERT the sequence
        self._conn.execute(
            """INSERT INTO document_sequences (project_id, doc_type_code, last_sequence, updated_at)
               VALUES (?, ?, 1, ?)
               ON CONFLICT(project_id, doc_type_code)
               DO UPDATE SET last_sequence = last_sequence + 1, updated_at = ?""",
            (project_id, doc_type_code.upper(), now, now),
        )
        self._conn.commit()
        cur = self._conn.execute(
            "SELECT last_sequence FROM document_sequences WHERE project_id = ? AND doc_type_code = ?",
            (project_id, doc_type_code.upper()),
        )
        seq = cur.fetchone()[0]
        proj_num = project['project_number'] + (project.get('variant') or '')
        return f"{proj_num}-{doc_type_code.upper()}-{seq:03d}"

    def get_doc_sequences(self, project_id):
        """List all sequences for a project."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT * FROM document_sequences WHERE project_id = ? ORDER BY doc_type_code",
            (project_id,),
        )
        rows = [dict(row) for row in cur.fetchall()]
        self._conn.row_factory = None
        return rows

    def seed_doc_sequences_from_scan(self, project_id, detected):
        """Set counters from filesystem scan (only if higher than current).

        detected: dict of {doc_type_code: max_sequence_found}
        """
        now = datetime.now().isoformat()
        for code, max_seq in detected.items():
            code = code.upper()
            cur = self._conn.execute(
                "SELECT last_sequence FROM document_sequences WHERE project_id = ? AND doc_type_code = ?",
                (project_id, code),
            )
            row = cur.fetchone()
            if row:
                if max_seq > row[0]:
                    self._conn.execute(
                        "UPDATE document_sequences SET last_sequence = ?, updated_at = ? WHERE project_id = ? AND doc_type_code = ?",
                        (max_seq, now, project_id, code),
                    )
            else:
                self._conn.execute(
                    "INSERT INTO document_sequences (project_id, doc_type_code, last_sequence, updated_at) VALUES (?, ?, ?, ?)",
                    (project_id, code, max_seq, now),
                )
        self._conn.commit()

    def close(self):
        self._conn.close()
