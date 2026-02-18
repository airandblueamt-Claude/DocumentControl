"""Pluggable email classifier for document control scope detection and categorization.

Supports rule-based (keyword matching), Claude API, Gemini, and Groq (AI-powered) classification.
Includes a FallbackChainClassifier for automatic failover.
"""

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when an API returns a rate-limit (429) response.

    Propagated through the FallbackChainClassifier so it can set a cooldown
    on the offending classifier and immediately try the next one.
    """
    pass


def _is_rate_limit_error(exc):
    """Detect whether an exception is a rate-limit / 429 error.

    Checks exception type names and message text for common patterns across
    Gemini (google.api_core.exceptions.ResourceExhausted),
    Groq (groq.RateLimitError), and Anthropic (anthropic.RateLimitError).
    """
    # Check class name (works without importing the SDK)
    cls_name = type(exc).__name__
    if cls_name in ('RateLimitError', 'ResourceExhausted', 'TooManyRequests'):
        return True
    # Check status code attribute (many HTTP SDKs set this)
    status = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
    if status == 429:
        return True
    # Check message text
    msg = str(exc).lower()
    if any(phrase in msg for phrase in ('429', 'rate limit', 'resource_exhausted',
                                        'too many requests', 'quota exceeded')):
        return True
    return False

# File extensions that indicate document attachments
DOC_EXTENSIONS = {'.pdf', '.docx', '.xlsx', '.dwg', '.dxf', '.msg', '.zip', '.rar'}


class ClassificationResult:
    """Value object holding classification output."""

    __slots__ = (
        'in_scope', 'doc_type', 'discipline', 'department',
        'response_required', 'references', 'confidence',
        'summary', 'priority',
    )

    def __init__(self, in_scope=True, doc_type='Others', discipline='General',
                 department='', response_required=False, references=None,
                 confidence=1.0, summary='', priority='medium'):
        self.in_scope = in_scope
        self.doc_type = doc_type
        self.discipline = discipline
        self.department = department
        self.response_required = response_required
        self.references = references or []
        self.confidence = confidence
        self.summary = summary
        self.priority = priority

    def to_dict(self):
        return {
            'in_scope': self.in_scope,
            'doc_type': self.doc_type,
            'discipline': self.discipline,
            'department': self.department,
            'response_required': self.response_required,
            'references': self.references,
            'confidence': self.confidence,
            'summary': self.summary,
            'priority': self.priority,
        }


class EmailClassifier(ABC):
    """Abstract base class for email classifiers."""

    @abstractmethod
    def is_in_scope(self, msg_data):
        """Return True if the email is related to document control."""

    @abstractmethod
    def classify(self, msg_data):
        """Return a ClassificationResult for the given email."""

    def classify_full(self, msg_data):
        """Combined scope + classify in one call. Default calls both separately.
        LLM subclasses override to use a single API call."""
        if not self.is_in_scope(msg_data):
            return ClassificationResult(
                in_scope=False, summary='Out of scope', priority='low', confidence=0.0)
        result = self.classify(msg_data)
        return result

    def classify_batch(self, email_list):
        """Classify multiple emails. Default falls back to individual classify_full() calls."""
        return [self.classify_full(msg) for msg in email_list]


class RuleBasedClassifier(EmailClassifier):
    """Keyword-matching classifier using config patterns and department keywords from SQLite."""

    @property
    def name(self):
        return "Keywords"

    def __init__(self, config, departments=None, custom_instructions=None, contacts=None):
        """
        Args:
            config: classification_config dict (from YAML)
            departments: list of department dicts [{name, keywords (JSON string), ...}]
            custom_instructions: ignored for rule-based (kept for uniform signature)
            contacts: list of contact dicts [{email, department, company, ...}]
        """
        self.scope_keywords = [kw.lower() for kw in config.get('scope_keywords', [])]
        self.ref_regexes = [re.compile(p) for p in config.get('ref_regexes', [])]
        self.type_keywords = config.get('type_keywords', {})
        self.discipline_keywords = config.get('discipline_keywords', {})
        self.response_phrases = [
            p.lower() for p in config.get('response_required_phrases', [])
        ]

        # Build department lookup from SQLite data
        self.departments = {}
        if departments:
            for dept in departments:
                name = dept['name']
                kw_raw = dept.get('keywords', '[]')
                if isinstance(kw_raw, str):
                    try:
                        keywords = [k.lower() for k in json.loads(kw_raw)]
                    except (ValueError, TypeError):
                        keywords = []
                else:
                    keywords = [k.lower() for k in kw_raw]
                self.departments[name] = keywords

        # Build contact email → department lookup
        self.contact_map = {}  # email.lower() → department
        if contacts:
            for c in contacts:
                email = (c.get('email') or '').strip().lower()
                dept = (c.get('department') or '').strip()
                if email:
                    self.contact_map[email] = dept

    def _get_text(self, msg_data):
        """Combine subject + body for keyword matching."""
        return f"{msg_data.get('subject', '')} {msg_data.get('body', '')}".lower()

    def _has_doc_attachments(self, msg_data):
        """Check if any attachment has a document-like extension."""
        for att in msg_data.get('attachments', []):
            filename = att.get('filename', '')
            ext = os.path.splitext(filename)[1].lower()
            if ext in DOC_EXTENSIONS:
                return True
        return False

    def is_in_scope(self, msg_data):
        """Scope check: known contact OR scope_keywords OR ref patterns OR type_keywords OR doc attachments."""
        # Known contacts are always in scope
        sender = (msg_data.get('sender') or '').strip().lower()
        if sender and sender in self.contact_map:
            return True

        text = self._get_text(msg_data)

        # Check scope keywords
        for kw in self.scope_keywords:
            if kw in text:
                return True

        # Check reference regex matches
        full_text = f"{msg_data.get('subject', '')} {msg_data.get('body', '')}"
        for regex in self.ref_regexes:
            if regex.search(full_text):
                return True

        # Check type keywords
        for doc_type, keywords in self.type_keywords.items():
            for kw in keywords:
                if kw.lower() in text:
                    return True

        # Check for document attachments
        if self._has_doc_attachments(msg_data):
            return True

        return False

    def classify(self, msg_data):
        """Full classification: type, discipline, department, response, references."""
        text = self._get_text(msg_data)

        # Document type
        doc_type = 'Others'
        for dtype, keywords in self.type_keywords.items():
            for kw in keywords:
                if kw.lower() in text:
                    doc_type = dtype
                    break
            if doc_type != 'Others':
                break

        # Discipline
        discipline = 'General'
        for disc, keywords in self.discipline_keywords.items():
            for kw in keywords:
                if kw.lower() in text:
                    discipline = disc
                    break
            if discipline != 'General':
                break

        # Department — check sender against contacts first
        department = ''
        sender = (msg_data.get('sender') or '').strip().lower()
        if sender and sender in self.contact_map and self.contact_map[sender]:
            department = self.contact_map[sender]

        # Fall back to keyword matching if no contact match
        if not department:
            for dept_name, keywords in self.departments.items():
                for kw in keywords:
                    if kw in text:
                        department = dept_name
                        break
                if department:
                    break

        # Response required
        response_required = False
        for phrase in self.response_phrases:
            if phrase in text:
                response_required = True
                break

        # Reference extraction
        references = []
        full_text = f"{msg_data.get('subject', '')} {msg_data.get('body', '')}"
        for regex in self.ref_regexes:
            references.extend(regex.findall(full_text))

        return ClassificationResult(
            in_scope=True,
            doc_type=doc_type,
            discipline=discipline,
            department=department,
            response_required=response_required,
            references=references,
            confidence=1.0,
        )

    def classify_full(self, msg_data):
        """Combined scope + classify for rule-based. Generates a simple summary."""
        if not self.is_in_scope(msg_data):
            return ClassificationResult(
                in_scope=False, summary='Out of scope', priority='low', confidence=0.0)
        result = self.classify(msg_data)
        # Generate simple summary from sender + subject
        sender = msg_data.get('sender_name') or msg_data.get('sender', '')
        subject = msg_data.get('subject', '')
        result.summary = f"{sender}: {subject}" if sender else subject
        result.priority = 'medium'
        return result


# ---------------------------------------------------------------------------
# LLM Base Class
# ---------------------------------------------------------------------------

class LLMClassifierBase(EmailClassifier):
    """Shared logic for all LLM-based classifiers (prompt building, email summary, fallback)."""

    def __init__(self, config, departments=None, custom_instructions=None, contacts=None):
        self._fallback = RuleBasedClassifier(config, departments, contacts=contacts)

        self._scope_keywords = config.get('scope_keywords', [])
        self._type_keywords = config.get('type_keywords', {})
        self._discipline_keywords = config.get('discipline_keywords', {})

        self._dept_names = []
        self._dept_list = departments or []
        if departments:
            self._dept_names = [d['name'] for d in departments]

        self._contacts = contacts or []
        self._custom_instructions = custom_instructions or ''
        self._max_body_chars = 2000
        self._fallback_on_error = True

    @staticmethod
    def _extract_text(resp):
        """Safely extract text from an LLM response, handling dict/object parts."""
        # Try .text first (works for most responses)
        try:
            return resp.text.strip()
        except (TypeError, AttributeError):
            pass
        # Fallback: walk candidates → parts and extract text manually
        try:
            parts = resp.candidates[0].content.parts
            texts = []
            for p in parts:
                if isinstance(p, str):
                    texts.append(p)
                elif hasattr(p, 'text'):
                    texts.append(p.text)
                elif isinstance(p, dict) and 'text' in p:
                    texts.append(p['text'])
            return ''.join(texts).strip()
        except (IndexError, KeyError, AttributeError, TypeError):
            raise ValueError(f"Cannot extract text from response: {resp}")

    def _email_summary(self, msg_data, body_limit=None):
        """Build a text summary of the email for the prompt."""
        if body_limit is None:
            body_limit = self._max_body_chars
        sender = msg_data.get('sender', '')
        sender_name = msg_data.get('sender_name', '')
        subject = msg_data.get('subject', '')
        cc = msg_data.get('cc', [])
        body = msg_data.get('body', '')[:body_limit]
        filenames = [a.get('filename', '') for a in msg_data.get('attachments', [])]

        from_str = f"{sender_name} <{sender}>" if sender_name else sender
        parts = [f"From: {from_str}", f"Subject: {subject}"]
        if cc:
            parts.append(f"CC: {', '.join(cc)}")
        parts.append(f"Body:\n{body}")
        if filenames:
            parts.append(f"Attachments: {', '.join(filenames)}")
        return '\n'.join(parts)

    def _build_scope_system_prompt(self):
        prompt = (
            "You are a document control assistant. Determine if the following email "
            "is related to document control, construction project management, or "
            "engineering correspondence.\n\n"
            f"Scope keywords for reference: {', '.join(self._scope_keywords)}\n\n"
        )

        # Add known contacts — emails from known contacts are always in scope
        if self._contacts:
            contact_lines = [f"- {c.get('email', '')} ({c.get('name', '')}, {c.get('company', '')})"
                             for c in self._contacts if c.get('email')]
            prompt += (
                "Known project contacts (emails from these senders are ALWAYS in scope):\n"
                + '\n'.join(contact_lines) + "\n\n"
            )

        prompt += "Reply with ONLY valid JSON: {\"in_scope\": true/false, \"reason\": \"brief explanation\"}"
        return prompt

    def _build_classify_system_prompt(self):
        doc_types = list(self._type_keywords.keys())
        disciplines = list(self._discipline_keywords.keys())

        prompt = (
            "You are a document control classifier for a construction/engineering company.\n"
            "Classify the following email.\n\n"
        )

        if self._custom_instructions:
            prompt += (
                "--- CUSTOM INSTRUCTIONS ---\n"
                f"{self._custom_instructions}\n"
                "--- END CUSTOM INSTRUCTIONS ---\n\n"
            )

        prompt += self._build_contacts_block()

        prompt += (
            f"Valid document types: {', '.join(doc_types)}\n"
            f"Valid disciplines: {', '.join(disciplines)}\n\n"
            f"Departments:\n{self._build_dept_block()}\n\n"
            "Reply with ONLY valid JSON:\n"
            "{\n"
            '  "doc_type": "one of the valid types above",\n'
            '  "discipline": "one of the valid disciplines above",\n'
            '  "department": "one of the valid departments above or empty string",\n'
            '  "response_required": true or false,\n'
            '  "references": ["any reference numbers found"],\n'
            '  "confidence": 0.0 to 1.0\n'
            "}"
        )
        return prompt

    def _build_dept_block(self):
        """Build department descriptions block for prompts."""
        dept_lines = []
        for dept in self._dept_list:
            name = dept['name']
            desc = dept.get('description', '') or ''
            if not desc:
                kw_raw = dept.get('keywords', '[]')
                if isinstance(kw_raw, str):
                    try:
                        kws = json.loads(kw_raw)
                    except (ValueError, TypeError):
                        kws = []
                else:
                    kws = kw_raw
                desc = ', '.join(kws) if kws else ''
            dept_lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        return '\n'.join(dept_lines) if dept_lines else 'none'

    def _build_contacts_block(self):
        """Build contacts block for prompts."""
        if not self._contacts:
            return ''
        contact_lines = []
        for c in self._contacts:
            email = c.get('email', '')
            name = c.get('name', '')
            dept = c.get('department', '')
            company = c.get('company', '')
            line = f"- {email} → {name}"
            if company:
                line += f" ({company})"
            if dept:
                line += f" → Department: {dept}"
            contact_lines.append(line)
        return (
            "Known contacts (use these to determine department):\n"
            + '\n'.join(contact_lines) + "\n\n"
        )

    def _build_full_system_prompt(self):
        """Combined scope + classify prompt for a single API call."""
        doc_types = list(self._type_keywords.keys())
        disciplines = list(self._discipline_keywords.keys())

        prompt = (
            "You are a document control assistant for a construction/engineering company.\n"
            "Determine if the following email is in scope (related to document control, "
            "construction project management, or engineering correspondence), "
            "and if so, classify it.\n\n"
            f"Scope keywords for reference: {', '.join(self._scope_keywords)}\n\n"
        )

        if self._custom_instructions:
            prompt += (
                "--- CUSTOM INSTRUCTIONS ---\n"
                f"{self._custom_instructions}\n"
                "--- END CUSTOM INSTRUCTIONS ---\n\n"
            )

        # Known contacts
        if self._contacts:
            scope_contacts = [f"- {c.get('email', '')} ({c.get('name', '')}, {c.get('company', '')})"
                              for c in self._contacts if c.get('email')]
            prompt += (
                "Known project contacts (emails from these senders are ALWAYS in scope):\n"
                + '\n'.join(scope_contacts) + "\n\n"
            )

        prompt += self._build_contacts_block()

        prompt += (
            f"Valid document types: {', '.join(doc_types)}\n"
            f"Valid disciplines: {', '.join(disciplines)}\n\n"
            f"Departments:\n{self._build_dept_block()}\n\n"
            "Reply with ONLY valid JSON:\n"
            "{\n"
            '  "in_scope": true or false,\n'
            '  "doc_type": "one of the valid types above (or empty if out of scope)",\n'
            '  "discipline": "one of the valid disciplines above (or empty if out of scope)",\n'
            '  "department": "one of the valid departments above or empty string",\n'
            '  "response_required": true or false,\n'
            '  "references": ["any reference numbers found"],\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "summary": "1-2 sentence summary of what this email is about",\n'
            '  "priority": "high" or "medium" or "low"\n'
            "}"
        )
        return prompt

    def _parse_full_response(self, text):
        """Parse combined scope+classify JSON response into ClassificationResult."""
        data = json.loads(text)
        in_scope = bool(data.get('in_scope', True))
        logger.info("LLM classify_full → in_scope=%s, type=%s, dept=%s, conf=%.2f, priority=%s",
                     in_scope, data.get('doc_type'), data.get('department'),
                     data.get('confidence', 0), data.get('priority', 'medium'))
        return ClassificationResult(
            in_scope=in_scope,
            doc_type=data.get('doc_type', 'Others') if in_scope else '',
            discipline=data.get('discipline', 'General') if in_scope else '',
            department=data.get('department', ''),
            response_required=bool(data.get('response_required', False)),
            references=data.get('references', []),
            confidence=float(data.get('confidence', 0.8)),
            summary=data.get('summary', ''),
            priority=data.get('priority', 'medium'),
        )

    def _full_json_schema(self):
        """JSON schema for Gemini structured output (combined scope+classify)."""
        doc_types = [t for t in self._type_keywords.keys() if t]
        disciplines = [d for d in self._discipline_keywords.keys() if d]
        return {
            "type": "object",
            "properties": {
                "in_scope": {"type": "boolean"},
                "doc_type": {"type": "string", "enum": doc_types},
                "discipline": {"type": "string", "enum": disciplines},
                "department": {"type": "string"},
                "response_required": {"type": "boolean"},
                "references": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "summary": {"type": "string"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["in_scope", "doc_type", "discipline", "department",
                         "response_required", "references", "confidence",
                         "summary", "priority"],
        }

    def _build_batch_system_prompt(self, count):
        """Prompt for classifying N emails in one API call."""
        doc_types = list(self._type_keywords.keys())
        disciplines = list(self._discipline_keywords.keys())

        prompt = (
            "You are a document control assistant for a construction/engineering company.\n"
            f"You will receive {count} emails below. For EACH email, determine if it is in scope "
            "and classify it.\n\n"
            f"Scope keywords for reference: {', '.join(self._scope_keywords)}\n\n"
        )

        if self._custom_instructions:
            prompt += (
                "--- CUSTOM INSTRUCTIONS ---\n"
                f"{self._custom_instructions}\n"
                "--- END CUSTOM INSTRUCTIONS ---\n\n"
            )

        if self._contacts:
            scope_contacts = [f"- {c.get('email', '')} ({c.get('name', '')}, {c.get('company', '')})"
                              for c in self._contacts if c.get('email')]
            prompt += (
                "Known project contacts (emails from these senders are ALWAYS in scope):\n"
                + '\n'.join(scope_contacts) + "\n\n"
            )

        prompt += self._build_contacts_block()

        prompt += (
            f"Valid document types: {', '.join(doc_types)}\n"
            f"Valid disciplines: {', '.join(disciplines)}\n\n"
            f"Departments:\n{self._build_dept_block()}\n\n"
            f"Reply with ONLY a valid JSON ARRAY of exactly {count} objects, one per email in order:\n"
            "[\n"
            "  {\n"
            '    "in_scope": true or false,\n'
            '    "doc_type": "...", "discipline": "...", "department": "...",\n'
            '    "response_required": true or false,\n'
            '    "references": ["..."],\n'
            '    "confidence": 0.0 to 1.0,\n'
            '    "summary": "1-2 sentence summary",\n'
            '    "priority": "high" or "medium" or "low"\n'
            "  },\n"
            "  ...\n"
            "]"
        )
        return prompt

    def _build_batch_user_prompt(self, email_list):
        """Format N emails for batch classification."""
        parts = []
        for i, msg in enumerate(email_list, 1):
            parts.append(f"--- EMAIL {i} ---")
            parts.append(self._email_summary(msg))
        return '\n\n'.join(parts)

    def _parse_batch_response(self, text, count):
        """Parse JSON array of classifications, pad/trim to expected count."""
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array for batch response")
        results = []
        for item in data[:count]:
            in_scope = bool(item.get('in_scope', True))
            results.append(ClassificationResult(
                in_scope=in_scope,
                doc_type=item.get('doc_type', 'Others') if in_scope else '',
                discipline=item.get('discipline', 'General') if in_scope else '',
                department=item.get('department', ''),
                response_required=bool(item.get('response_required', False)),
                references=item.get('references', []),
                confidence=float(item.get('confidence', 0.8)),
                summary=item.get('summary', ''),
                priority=item.get('priority', 'medium'),
            ))
        # Pad if LLM returned fewer than expected
        while len(results) < count:
            results.append(ClassificationResult(
                in_scope=True, confidence=0.5,
                summary='Classification missing from batch', priority='medium'))
        return results

    def _parse_scope_response(self, text):
        data = json.loads(text)
        logger.info("LLM is_in_scope → %s (reason: %s)", data['in_scope'], data.get('reason', ''))
        return bool(data['in_scope'])

    def _parse_classify_response(self, text):
        data = json.loads(text)
        logger.info("LLM classify → type=%s, disc=%s, dept=%s, conf=%.2f",
                     data.get('doc_type'), data.get('discipline'),
                     data.get('department'), data.get('confidence', 0))
        return ClassificationResult(
            in_scope=True,
            doc_type=data.get('doc_type', 'Others'),
            discipline=data.get('discipline', 'General'),
            department=data.get('department', ''),
            response_required=bool(data.get('response_required', False)),
            references=data.get('references', []),
            confidence=float(data.get('confidence', 0.8)),
        )


# ---------------------------------------------------------------------------
# Claude API Classifier
# ---------------------------------------------------------------------------

class ClaudeAPIClassifier(LLMClassifierBase):
    """AI-powered classifier using the Claude API with rule-based fallback."""

    @property
    def name(self):
        return "Claude"

    def __init__(self, config, departments=None, custom_instructions=None, contacts=None):
        super().__init__(config, departments, custom_instructions, contacts=contacts)
        import anthropic as _anthropic
        self._anthropic = _anthropic

        claude_cfg = config.get('classifier', {}).get('claude_api', {})
        api_key_env = claude_cfg.get('api_key_env', 'ANTHROPIC_API_KEY')
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{api_key_env}' is not set. "
                f"Set it to your Anthropic API key to use the Claude API classifier."
            )

        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model = claude_cfg.get('model', 'claude-sonnet-4-5-20250929')
        self._max_body_chars = claude_cfg.get('max_body_chars', 2000)
        self._fallback_on_error = claude_cfg.get('fallback_on_error', True)

    def is_in_scope(self, msg_data):
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=150,
                system=self._build_scope_system_prompt(),
                messages=[{"role": "user", "content": self._email_summary(msg_data, body_limit=500)}],
            )
            return self._parse_scope_response(resp.content[0].text.strip())
        except (self._anthropic.APIError, json.JSONDecodeError, KeyError, IndexError) as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.warning("Claude is_in_scope failed (%s), falling back to rule-based", exc)
            if self._fallback_on_error:
                return self._fallback.is_in_scope(msg_data)
            raise

    def classify(self, msg_data):
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=300,
                system=self._build_classify_system_prompt(),
                messages=[{"role": "user", "content": self._email_summary(msg_data)}],
            )
            return self._parse_classify_response(resp.content[0].text.strip())
        except (self._anthropic.APIError, json.JSONDecodeError, KeyError, IndexError) as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.warning("Claude classify failed (%s), falling back to rule-based", exc)
            if self._fallback_on_error:
                return self._fallback.classify(msg_data)
            raise

    def classify_full(self, msg_data):
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=500,
                system=self._build_full_system_prompt(),
                messages=[{"role": "user", "content": self._email_summary(msg_data)}],
            )
            return self._parse_full_response(resp.content[0].text.strip())
        except (self._anthropic.APIError, json.JSONDecodeError, KeyError, IndexError) as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.warning("Claude classify_full failed (%s), falling back", exc)
            if self._fallback_on_error:
                return self._fallback.classify_full(msg_data)
            raise

    def classify_batch(self, email_list):
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=300 * len(email_list),
                system=self._build_batch_system_prompt(len(email_list)),
                messages=[{"role": "user", "content": self._build_batch_user_prompt(email_list)}],
            )
            return self._parse_batch_response(resp.content[0].text.strip(), len(email_list))
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.warning("Claude classify_batch failed (%s), falling back to individual", exc)
            return [self.classify_full(msg) for msg in email_list]


# ---------------------------------------------------------------------------
# Gemini Classifier
# ---------------------------------------------------------------------------

class GeminiClassifier(LLMClassifierBase):
    """AI-powered classifier using Google Gemini (free tier) with rule-based fallback."""

    @property
    def name(self):
        return "Gemini"

    def __init__(self, config, departments=None, custom_instructions=None, contacts=None):
        super().__init__(config, departments, custom_instructions, contacts=contacts)
        from google import genai as _genai
        self._genai = _genai

        gemini_cfg = config.get('classifier', {}).get('gemini', {})
        api_key_env = gemini_cfg.get('api_key_env', 'GEMINI_API_KEY')
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{api_key_env}' is not set. "
                f"Set it to your Google AI API key to use the Gemini classifier."
            )

        self._client = _genai.Client(api_key=api_key)
        self._model = gemini_cfg.get('model', 'gemini-2.5-flash-lite')
        self._max_body_chars = gemini_cfg.get('max_body_chars', 2000)
        self._fallback_on_error = gemini_cfg.get('fallback_on_error', True)

    def _scope_json_schema(self):
        return {
            "type": "object",
            "properties": {
                "in_scope": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["in_scope", "reason"],
        }

    def _classify_json_schema(self):
        doc_types = [t for t in self._type_keywords.keys() if t]
        disciplines = [d for d in self._discipline_keywords.keys() if d]
        return {
            "type": "object",
            "properties": {
                "doc_type": {"type": "string", "enum": doc_types},
                "discipline": {"type": "string", "enum": disciplines},
                "department": {"type": "string"},
                "response_required": {"type": "boolean"},
                "references": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
            "required": ["doc_type", "discipline", "department", "response_required",
                         "references", "confidence"],
        }

    def is_in_scope(self, msg_data):
        try:
            from google.genai import types as gtypes
            resp = self._client.models.generate_content(
                model=self._model,
                contents=self._build_scope_system_prompt() + "\n\n" + self._email_summary(msg_data, body_limit=500),
                config=gtypes.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=self._scope_json_schema(),
                ),
            )
            return self._parse_scope_response(self._extract_text(resp))
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.warning("Gemini is_in_scope failed (%s), falling back to rule-based", exc)
            if self._fallback_on_error:
                return self._fallback.is_in_scope(msg_data)
            raise

    def classify(self, msg_data):
        try:
            from google.genai import types as gtypes
            resp = self._client.models.generate_content(
                model=self._model,
                contents=self._build_classify_system_prompt() + "\n\n" + self._email_summary(msg_data),
                config=gtypes.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=self._classify_json_schema(),
                ),
            )
            return self._parse_classify_response(self._extract_text(resp))
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.warning("Gemini classify failed (%s), falling back to rule-based", exc)
            if self._fallback_on_error:
                return self._fallback.classify(msg_data)
            raise

    def classify_full(self, msg_data):
        try:
            from google.genai import types as gtypes
            resp = self._client.models.generate_content(
                model=self._model,
                contents=self._build_full_system_prompt() + "\n\n" + self._email_summary(msg_data),
                config=gtypes.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=self._full_json_schema(),
                ),
            )
            return self._parse_full_response(self._extract_text(resp))
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.warning("Gemini classify_full failed (%s), falling back", exc)
            if self._fallback_on_error:
                return self._fallback.classify_full(msg_data)
            raise

    def classify_batch(self, email_list):
        try:
            from google.genai import types as gtypes
            resp = self._client.models.generate_content(
                model=self._model,
                contents=(self._build_batch_system_prompt(len(email_list))
                          + "\n\n" + self._build_batch_user_prompt(email_list)),
                config=gtypes.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            return self._parse_batch_response(self._extract_text(resp), len(email_list))
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.warning("Gemini classify_batch failed (%s), falling back to individual", exc)
            return [self.classify_full(msg) for msg in email_list]


# ---------------------------------------------------------------------------
# Groq Classifier
# ---------------------------------------------------------------------------

class GroqClassifier(LLMClassifierBase):
    """AI-powered classifier using Groq (free tier Llama) with rule-based fallback."""

    @property
    def name(self):
        return "Groq"

    def __init__(self, config, departments=None, custom_instructions=None, contacts=None):
        super().__init__(config, departments, custom_instructions, contacts=contacts)
        import groq as _groq
        self._groq = _groq

        groq_cfg = config.get('classifier', {}).get('groq', {})
        api_key_env = groq_cfg.get('api_key_env', 'GROQ_API_KEY')
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{api_key_env}' is not set. "
                f"Set it to your Groq API key to use the Groq classifier."
            )

        self._client = _groq.Groq(api_key=api_key)
        self._model = groq_cfg.get('model', 'llama-3.3-70b-versatile')
        self._max_body_chars = groq_cfg.get('max_body_chars', 2000)
        self._fallback_on_error = groq_cfg.get('fallback_on_error', True)

    def is_in_scope(self, msg_data):
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._build_scope_system_prompt()},
                    {"role": "user", "content": self._email_summary(msg_data, body_limit=500)},
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
            )
            return self._parse_scope_response(resp.choices[0].message.content.strip())
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.warning("Groq is_in_scope failed (%s), falling back to rule-based", exc)
            if self._fallback_on_error:
                return self._fallback.is_in_scope(msg_data)
            raise

    def classify(self, msg_data):
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._build_classify_system_prompt()},
                    {"role": "user", "content": self._email_summary(msg_data)},
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
            )
            return self._parse_classify_response(resp.choices[0].message.content.strip())
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.warning("Groq classify failed (%s), falling back to rule-based", exc)
            if self._fallback_on_error:
                return self._fallback.classify(msg_data)
            raise

    def classify_full(self, msg_data):
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._build_full_system_prompt()},
                    {"role": "user", "content": self._email_summary(msg_data)},
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
            )
            return self._parse_full_response(resp.choices[0].message.content.strip())
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.warning("Groq classify_full failed (%s), falling back", exc)
            if self._fallback_on_error:
                return self._fallback.classify_full(msg_data)
            raise

    def classify_batch(self, email_list):
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._build_batch_system_prompt(len(email_list))},
                    {"role": "user", "content": self._build_batch_user_prompt(email_list)},
                ],
                response_format={"type": "json_object"},
                max_tokens=300 * len(email_list),
            )
            return self._parse_batch_response(resp.choices[0].message.content.strip(), len(email_list))
        except Exception as exc:
            if _is_rate_limit_error(exc):
                raise RateLimitError(str(exc)) from exc
            logger.warning("Groq classify_batch failed (%s), falling back to individual", exc)
            return [self.classify_full(msg) for msg in email_list]


# ---------------------------------------------------------------------------
# Fallback Chain Classifier
# ---------------------------------------------------------------------------

class FallbackChainClassifier(EmailClassifier):
    """Wraps an ordered list of classifiers, tries each in sequence.

    On exception, logs warning and moves to next.
    On RateLimitError, sets a 60-second cooldown so subsequent calls skip
    the rate-limited classifier instead of wasting API calls.
    If all fail, defaults to in_scope=True (safer to review than miss).
    """

    RATE_LIMIT_COOLDOWN = 60  # seconds to skip a rate-limited classifier

    def __init__(self, classifiers):
        """
        Args:
            classifiers: ordered list of EmailClassifier instances
        """
        self._classifiers = classifiers
        self._last_used_name = 'Unknown'
        # Maps classifier index → time.time() when cooldown expires
        self._cooldowns = {}

    @property
    def name(self):
        return self._last_used_name

    def _is_cooled_down(self, idx):
        """Check if classifier at index is in rate-limit cooldown."""
        expiry = self._cooldowns.get(idx)
        if expiry is None:
            return False
        if time.time() >= expiry:
            del self._cooldowns[idx]
            return False
        return True

    def _set_cooldown(self, idx, clf):
        """Set a cooldown on classifier at index after a rate-limit error."""
        self._cooldowns[idx] = time.time() + self.RATE_LIMIT_COOLDOWN
        name = getattr(clf, 'name', clf.__class__.__name__)
        logger.warning("FallbackChain: %s rate-limited, skipping for %ds",
                        name, self.RATE_LIMIT_COOLDOWN)

    def _try_classifiers(self, method_name, *args, **kwargs):
        """Generic helper to try each classifier in order with cooldown awareness.

        Returns the result from the first successful classifier.
        Raises _AllFailed if none succeed.
        """
        for idx, clf in enumerate(self._classifiers):
            if self._is_cooled_down(idx):
                name = getattr(clf, 'name', clf.__class__.__name__)
                logger.debug("FallbackChain: skipping %s (rate-limit cooldown)", name)
                continue
            try:
                method = getattr(clf, method_name)
                result = method(*args, **kwargs)
                self._last_used_name = getattr(clf, 'name', clf.__class__.__name__)
                return result
            except RateLimitError as exc:
                self._set_cooldown(idx, clf)
            except Exception as exc:
                logger.warning("FallbackChain: %s.%s failed (%s), trying next",
                               clf.__class__.__name__, method_name, exc)
        return None  # sentinel — caller handles the "all failed" case

    def is_in_scope(self, msg_data):
        result = self._try_classifiers('is_in_scope', msg_data)
        if result is not None:
            return result
        logger.warning("FallbackChain: all classifiers failed for is_in_scope, defaulting to True")
        return True

    def classify(self, msg_data):
        result = self._try_classifiers('classify', msg_data)
        if result is not None:
            return result
        logger.warning("FallbackChain: all classifiers failed for classify, returning defaults")
        return ClassificationResult()

    def classify_full(self, msg_data):
        result = self._try_classifiers('classify_full', msg_data)
        if result is not None:
            return result
        logger.warning("FallbackChain: all classifiers failed for classify_full, returning defaults")
        return ClassificationResult()

    def classify_batch(self, email_list):
        result = self._try_classifiers('classify_batch', email_list)
        if result is not None:
            return result
        logger.warning("FallbackChain: all classifiers failed for classify_batch, returning defaults")
        return [ClassificationResult() for _ in email_list]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_classifier(config, departments=None, method=None, custom_instructions=None, contacts=None):
    """Factory function to create the appropriate classifier.

    Args:
        config: classification_config dict
        departments: list of department dicts from SQLite
        method: 'rule_based', 'claude_api', 'gemini', 'groq', or 'auto'.
                Defaults to config value.
        custom_instructions: optional string with user-defined AI instructions
        contacts: list of contact dicts from SQLite
    """
    if method is None:
        method = config.get('classifier', {}).get('method', 'rule_based')

    if method == 'rule_based':
        return RuleBasedClassifier(config, departments, custom_instructions, contacts=contacts)
    elif method == 'claude_api':
        return ClaudeAPIClassifier(config, departments, custom_instructions, contacts=contacts)
    elif method == 'gemini':
        return GeminiClassifier(config, departments, custom_instructions, contacts=contacts)
    elif method == 'groq':
        return GroqClassifier(config, departments, custom_instructions, contacts=contacts)
    elif method == 'auto':
        return _build_auto_chain(config, departments, custom_instructions, contacts=contacts)
    else:
        raise ValueError(f"Unknown classifier method: {method}")


def _build_auto_chain(config, departments, custom_instructions=None, contacts=None):
    """Build a FallbackChainClassifier from whichever API keys are present."""
    chain = []

    # Try Gemini first (free tier)
    if os.environ.get('GEMINI_API_KEY'):
        try:
            chain.append(GeminiClassifier(config, departments, custom_instructions, contacts=contacts))
            logger.info("Auto classifier: Gemini enabled")
        except Exception as exc:
            logger.warning("Auto classifier: Gemini init failed (%s)", exc)

    # Then Groq (free tier)
    if os.environ.get('GROQ_API_KEY'):
        try:
            chain.append(GroqClassifier(config, departments, custom_instructions, contacts=contacts))
            logger.info("Auto classifier: Groq enabled")
        except Exception as exc:
            logger.warning("Auto classifier: Groq init failed (%s)", exc)

    # Then Claude (paid)
    if os.environ.get('ANTHROPIC_API_KEY'):
        try:
            chain.append(ClaudeAPIClassifier(config, departments, custom_instructions, contacts=contacts))
            logger.info("Auto classifier: Claude API enabled")
        except Exception as exc:
            logger.warning("Auto classifier: Claude API init failed (%s)", exc)

    # Always end with rule-based
    chain.append(RuleBasedClassifier(config, departments, custom_instructions, contacts=contacts))
    logger.info("Auto classifier: RuleBasedClassifier as final fallback")

    if len(chain) == 1:
        # Only rule-based, no need for chain wrapper
        return chain[0]

    return FallbackChainClassifier(chain)
