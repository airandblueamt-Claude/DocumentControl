"""Pluggable email classifier for document control scope detection and categorization.

Supports rule-based (keyword matching), Claude API, Gemini, and Groq (AI-powered) classification.
Includes a FallbackChainClassifier for automatic failover.
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# File extensions that indicate document attachments
DOC_EXTENSIONS = {'.pdf', '.docx', '.xlsx', '.dwg', '.dxf', '.msg', '.zip', '.rar'}


class ClassificationResult:
    """Value object holding classification output."""

    __slots__ = (
        'in_scope', 'doc_type', 'discipline', 'department',
        'response_required', 'references', 'confidence',
    )

    def __init__(self, in_scope=True, doc_type='Others', discipline='General',
                 department='', response_required=False, references=None,
                 confidence=1.0):
        self.in_scope = in_scope
        self.doc_type = doc_type
        self.discipline = discipline
        self.department = department
        self.response_required = response_required
        self.references = references or []
        self.confidence = confidence

    def to_dict(self):
        return {
            'in_scope': self.in_scope,
            'doc_type': self.doc_type,
            'discipline': self.discipline,
            'department': self.department,
            'response_required': self.response_required,
            'references': self.references,
            'confidence': self.confidence,
        }


class EmailClassifier(ABC):
    """Abstract base class for email classifiers."""

    @abstractmethod
    def is_in_scope(self, msg_data):
        """Return True if the email is related to document control."""

    @abstractmethod
    def classify(self, msg_data):
        """Return a ClassificationResult for the given email."""


class RuleBasedClassifier(EmailClassifier):
    """Keyword-matching classifier using config patterns and department keywords from SQLite."""

    @property
    def name(self):
        return "Keywords"

    def __init__(self, config, departments=None, custom_instructions=None):
        """
        Args:
            config: classification_config dict (from YAML)
            departments: list of department dicts [{name, keywords (JSON string), ...}]
            custom_instructions: ignored for rule-based (kept for uniform signature)
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
        """Scope check: matches scope_keywords OR ref patterns OR type_keywords OR doc attachments."""
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

        # Department (from SQLite departments)
        department = ''
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


# ---------------------------------------------------------------------------
# LLM Base Class
# ---------------------------------------------------------------------------

class LLMClassifierBase(EmailClassifier):
    """Shared logic for all LLM-based classifiers (prompt building, email summary, fallback)."""

    def __init__(self, config, departments=None, custom_instructions=None):
        self._fallback = RuleBasedClassifier(config, departments)

        self._scope_keywords = config.get('scope_keywords', [])
        self._type_keywords = config.get('type_keywords', {})
        self._discipline_keywords = config.get('discipline_keywords', {})

        self._dept_names = []
        self._dept_list = departments or []
        if departments:
            self._dept_names = [d['name'] for d in departments]

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
        return (
            "You are a document control assistant. Determine if the following email "
            "is related to document control, construction project management, or "
            "engineering correspondence.\n\n"
            f"Scope keywords for reference: {', '.join(self._scope_keywords)}\n\n"
            "Reply with ONLY valid JSON: {\"in_scope\": true/false, \"reason\": \"brief explanation\"}"
        )

    def _build_classify_system_prompt(self):
        doc_types = list(self._type_keywords.keys())
        disciplines = list(self._discipline_keywords.keys())

        # Build department descriptions
        dept_lines = []
        for dept in self._dept_list:
            name = dept['name']
            desc = dept.get('description', '') or ''
            if not desc:
                # Fall back to keywords list
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
        dept_block = '\n'.join(dept_lines) if dept_lines else 'none'

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

        prompt += (
            f"Valid document types: {', '.join(doc_types)}\n"
            f"Valid disciplines: {', '.join(disciplines)}\n\n"
            f"Departments:\n{dept_block}\n\n"
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

    def __init__(self, config, departments=None, custom_instructions=None):
        super().__init__(config, departments, custom_instructions)
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
            logger.warning("Claude classify failed (%s), falling back to rule-based", exc)
            if self._fallback_on_error:
                return self._fallback.classify(msg_data)
            raise


# ---------------------------------------------------------------------------
# Gemini Classifier
# ---------------------------------------------------------------------------

class GeminiClassifier(LLMClassifierBase):
    """AI-powered classifier using Google Gemini (free tier) with rule-based fallback."""

    @property
    def name(self):
        return "Gemini"

    def __init__(self, config, departments=None, custom_instructions=None):
        super().__init__(config, departments, custom_instructions)
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
        doc_types = list(self._type_keywords.keys())
        disciplines = list(self._discipline_keywords.keys())
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
            logger.warning("Gemini classify failed (%s), falling back to rule-based", exc)
            if self._fallback_on_error:
                return self._fallback.classify(msg_data)
            raise


# ---------------------------------------------------------------------------
# Groq Classifier
# ---------------------------------------------------------------------------

class GroqClassifier(LLMClassifierBase):
    """AI-powered classifier using Groq (free tier Llama) with rule-based fallback."""

    @property
    def name(self):
        return "Groq"

    def __init__(self, config, departments=None, custom_instructions=None):
        super().__init__(config, departments, custom_instructions)
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
            logger.warning("Groq classify failed (%s), falling back to rule-based", exc)
            if self._fallback_on_error:
                return self._fallback.classify(msg_data)
            raise


# ---------------------------------------------------------------------------
# Fallback Chain Classifier
# ---------------------------------------------------------------------------

class FallbackChainClassifier(EmailClassifier):
    """Wraps an ordered list of classifiers, tries each in sequence.

    On exception, logs warning and moves to next.
    If all fail, defaults to in_scope=True (safer to review than miss).
    """

    def __init__(self, classifiers):
        """
        Args:
            classifiers: ordered list of EmailClassifier instances
        """
        self._classifiers = classifiers
        self._last_used_name = 'Unknown'

    @property
    def name(self):
        return self._last_used_name

    def is_in_scope(self, msg_data):
        for clf in self._classifiers:
            try:
                result = clf.is_in_scope(msg_data)
                self._last_used_name = getattr(clf, 'name', clf.__class__.__name__)
                return result
            except Exception as exc:
                logger.warning("FallbackChain: %s.is_in_scope failed (%s), trying next",
                               clf.__class__.__name__, exc)
        logger.warning("FallbackChain: all classifiers failed for is_in_scope, defaulting to True")
        return True

    def classify(self, msg_data):
        for clf in self._classifiers:
            try:
                result = clf.classify(msg_data)
                self._last_used_name = getattr(clf, 'name', clf.__class__.__name__)
                return result
            except Exception as exc:
                logger.warning("FallbackChain: %s.classify failed (%s), trying next",
                               clf.__class__.__name__, exc)
        logger.warning("FallbackChain: all classifiers failed for classify, returning defaults")
        return ClassificationResult()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_classifier(config, departments=None, method=None, custom_instructions=None):
    """Factory function to create the appropriate classifier.

    Args:
        config: classification_config dict
        departments: list of department dicts from SQLite
        method: 'rule_based', 'claude_api', 'gemini', 'groq', or 'auto'.
                Defaults to config value.
        custom_instructions: optional string with user-defined AI instructions
    """
    if method is None:
        method = config.get('classifier', {}).get('method', 'rule_based')

    if method == 'rule_based':
        return RuleBasedClassifier(config, departments, custom_instructions)
    elif method == 'claude_api':
        return ClaudeAPIClassifier(config, departments, custom_instructions)
    elif method == 'gemini':
        return GeminiClassifier(config, departments, custom_instructions)
    elif method == 'groq':
        return GroqClassifier(config, departments, custom_instructions)
    elif method == 'auto':
        return _build_auto_chain(config, departments, custom_instructions)
    else:
        raise ValueError(f"Unknown classifier method: {method}")


def _build_auto_chain(config, departments, custom_instructions=None):
    """Build a FallbackChainClassifier from whichever API keys are present."""
    chain = []

    # Try Gemini first (free tier)
    if os.environ.get('GEMINI_API_KEY'):
        try:
            chain.append(GeminiClassifier(config, departments, custom_instructions))
            logger.info("Auto classifier: Gemini enabled")
        except Exception as exc:
            logger.warning("Auto classifier: Gemini init failed (%s)", exc)

    # Then Groq (free tier)
    if os.environ.get('GROQ_API_KEY'):
        try:
            chain.append(GroqClassifier(config, departments, custom_instructions))
            logger.info("Auto classifier: Groq enabled")
        except Exception as exc:
            logger.warning("Auto classifier: Groq init failed (%s)", exc)

    # Then Claude (paid)
    if os.environ.get('ANTHROPIC_API_KEY'):
        try:
            chain.append(ClaudeAPIClassifier(config, departments, custom_instructions))
            logger.info("Auto classifier: Claude API enabled")
        except Exception as exc:
            logger.warning("Auto classifier: Claude API init failed (%s)", exc)

    # Always end with rule-based
    chain.append(RuleBasedClassifier(config, departments, custom_instructions))
    logger.info("Auto classifier: RuleBasedClassifier as final fallback")

    if len(chain) == 1:
        # Only rule-based, no need for chain wrapper
        return chain[0]

    return FallbackChainClassifier(chain)
