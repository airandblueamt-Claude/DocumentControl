"""LLM-as-Judge module for evaluating classification quality.

Uses compound scoring with majority voting (inspired by Haize Labs Verdict):
- N cheap LLM calls per email, each independently evaluates the classification
- Majority vote determines final verdict per field
- Average of individual quality scores = overall score

Fully decoupled from scanner.py — never imported there.
"""

import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

NUM_JUDGE_FIELDS = 5  # len(JUDGE_FIELDS) — used in DB queries too

JUDGE_FIELDS = ['in_scope', 'doc_type', 'discipline', 'department', 'response_required']


@dataclass
class JudgeResult:
    """Result of judging a single classification."""
    quality_score: float = 0.5
    field_scores: dict = field(default_factory=dict)
    field_verdicts: dict = field(default_factory=dict)
    suggested_corrections: dict = field(default_factory=dict)
    reasoning: str = ''
    raw_responses: list = field(default_factory=list)
    vote_count: int = 1
    agree_count: int = 0
    error: str = ''


def _build_dept_context(departments):
    """Build department context string with keywords."""
    if not departments:
        return 'none configured'
    lines = []
    for d in departments:
        name = d['name']
        kw_raw = d.get('keywords', '[]')
        if isinstance(kw_raw, str):
            try:
                kws = json.loads(kw_raw)
            except (ValueError, TypeError):
                kws = []
        else:
            kws = kw_raw or []
        desc = d.get('description', '') or ''
        if desc:
            lines.append(f"  - {name}: {desc}")
        elif kws:
            lines.append(f"  - {name}: keywords={', '.join(kws)}")
        else:
            lines.append(f"  - {name}")
    return '\n'.join(lines)


def _build_contacts_context(contacts):
    """Build known contacts context."""
    if not contacts:
        return ''
    lines = []
    for c in contacts:
        email = c.get('email', '')
        name = c.get('name', '')
        dept = c.get('department', '')
        company = c.get('company', '')
        parts = [f"{email} ({name}"]
        if company:
            parts.append(f", {company}")
        parts.append(')')
        if dept:
            parts.append(f" -> dept: {dept}")
        lines.append('  - ' + ''.join(parts))
    return '\n'.join(lines)


def _build_judge_prompt(email_data, classification, config_context):
    """Build a domain-aware prompt for the judge LLM.

    config_context is a dict with:
      departments, contacts, doc_types, disciplines, scope_keywords,
      response_phrases, type_keywords, discipline_keywords
    """
    doc_types = config_context.get('doc_types', [])
    disciplines = config_context.get('disciplines', [])
    scope_keywords = config_context.get('scope_keywords', [])
    response_phrases = config_context.get('response_phrases', [])
    dept_text = config_context.get('dept_text', 'none')
    contacts_text = config_context.get('contacts_text', '')
    type_keywords = config_context.get('type_keywords', {})
    discipline_keywords = config_context.get('discipline_keywords', {})

    # Build type keyword hints
    type_hints = []
    for t, kws in type_keywords.items():
        type_hints.append(f"  - {t}: {', '.join(kws[:8])}")
    type_hint_text = '\n'.join(type_hints) if type_hints else '  (none)'

    disc_hints = []
    for d, kws in discipline_keywords.items():
        disc_hints.append(f"  - {d}: {', '.join(kws[:8])}")
    disc_hint_text = '\n'.join(disc_hints) if disc_hints else '  (none)'

    # Email body — use a meaningful excerpt
    body = (email_data.get('body', '') or '')
    # Skip forwarded headers and get actual content
    body_clean = body[:2000]

    sender = email_data.get('sender', '')
    sender_name = email_data.get('sender_name', '')
    subject = email_data.get('subject', '')

    prompt = f"""You are a quality judge for a construction/engineering document control system.
Your company manages correspondence for a KFUPM (King Fahd University) datacenter/ICT infrastructure project.

The system classifies incoming emails into document types, disciplines, and departments.
A classifier already assigned values to this email. You must evaluate if each assignment is CORRECT.

=== WHAT IS "IN SCOPE" ===
Emails are in scope if they relate to: document control, construction project management,
engineering correspondence, submittals, RFIs, drawings, specifications, site surveys, meeting minutes,
project coordination, or any technical/contractual project communication.

Emails that are NOT in scope: marketing emails, newsletters, security alerts, password resets,
personal emails, spam, social media notifications, software product announcements.

Scope keywords: {', '.join(scope_keywords)}
Known project contacts (emails from these are ALWAYS in scope):
{contacts_text or '  (none)'}

=== VALID DOCUMENT TYPES ===
{type_hint_text}

=== VALID DISCIPLINES ===
{disc_hint_text}

=== DEPARTMENTS ===
{dept_text}

=== RESPONSE REQUIRED PHRASES ===
{', '.join(response_phrases) if response_phrases else '(none)'}

=== EMAIL TO EVALUATE ===
From: {sender_name} <{sender}>
Subject: {subject}
Attachments: {email_data.get('attachment_count', 0)}
Body:
{body_clean}

=== CLASSIFICATION TO EVALUATE ===
doc_type: {classification.get('doc_type', '')}
discipline: {classification.get('discipline', '')}
department: {classification.get('department', '')}
response_required: {classification.get('response_required', False)}
(classified by: {classification.get('classifier_method', 'unknown')})

=== YOUR TASK ===
1. First decide: is this email actually IN SCOPE for document control? If NOT, score should be very low.
2. For each field, evaluate if the assigned value is correct given the email content.
3. For doc_type: does the content match the document type definition?
4. For discipline: does the subject matter match the discipline keywords?
5. For department: does the sender or content align with the department?
6. For response_required: does the email actually ask for a response?

Reply with ONLY valid JSON:
{{
  "quality_score": 0.0 to 1.0,
  "field_scores": {{
    "in_scope": 0.0-1.0 (1.0 if correctly identified as in/out of scope),
    "doc_type": 0.0-1.0,
    "discipline": 0.0-1.0,
    "department": 0.0-1.0,
    "response_required": 0.0-1.0
  }},
  "field_verdicts": {{
    "in_scope": "agree" or "disagree",
    "doc_type": "agree" or "disagree",
    "discipline": "agree" or "disagree",
    "department": "agree" or "disagree",
    "response_required": "agree" or "disagree"
  }},
  "suggested_corrections": {{"field_name": "suggested value"}},
  "reasoning": "1-2 sentence explanation"
}}"""
    return prompt


def _parse_judge_response(text):
    """Parse a single judge LLM response into a JudgeResult."""
    cleaned = text.strip()
    # Strip markdown code fences if present
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        cleaned = '\n'.join(lines).strip()

    data = json.loads(cleaned)

    field_verdicts = data.get('field_verdicts', {})
    agree_count = sum(1 for v in field_verdicts.values() if v == 'agree')

    return JudgeResult(
        quality_score=float(data.get('quality_score', 0.5)),
        field_scores=data.get('field_scores', {}),
        field_verdicts=field_verdicts,
        suggested_corrections=data.get('suggested_corrections', {}),
        reasoning=data.get('reasoning', ''),
        vote_count=1,
        agree_count=agree_count,
    )


def _aggregate_votes(results):
    """Aggregate multiple JudgeResults via majority voting."""
    if not results:
        return JudgeResult(reasoning='No votes collected', error='no_votes')

    valid = [r for r in results if not r.error]
    if not valid:
        return JudgeResult(
            reasoning='All votes failed',
            error='all_failed',
            raw_responses=[r.reasoning or r.error for r in results],
            vote_count=len(results),
            agree_count=0,
        )

    # Average quality score
    avg_score = sum(r.quality_score for r in valid) / len(valid)

    # Majority vote per field
    final_verdicts = {}
    final_field_scores = {}
    for f in JUDGE_FIELDS:
        verdicts = [r.field_verdicts.get(f, 'agree') for r in valid]
        counter = Counter(verdicts)
        final_verdicts[f] = counter.most_common(1)[0][0]
        scores = [r.field_scores.get(f, 0.5) for r in valid]
        final_field_scores[f] = round(sum(scores) / len(scores), 2)

    # Merge suggested corrections from disagreeing votes
    merged_corrections = {}
    for r in valid:
        for fld, val in r.suggested_corrections.items():
            if fld not in merged_corrections:
                merged_corrections[fld] = []
            merged_corrections[fld].append(val)
    final_corrections = {}
    for fld, suggestions in merged_corrections.items():
        if final_verdicts.get(fld) == 'disagree':
            counter = Counter(suggestions)
            final_corrections[fld] = counter.most_common(1)[0][0]

    # Count fields where majority agrees
    all_agree_count = sum(
        1 for f in JUDGE_FIELDS
        if final_verdicts.get(f) == 'agree'
    )

    # Combine reasoning
    reasons = [r.reasoning for r in valid if r.reasoning]
    combined_reasoning = ' | '.join(reasons[:3]) if reasons else ''

    return JudgeResult(
        quality_score=round(avg_score, 2),
        field_scores=final_field_scores,
        field_verdicts=final_verdicts,
        suggested_corrections=final_corrections,
        reasoning=combined_reasoning,
        raw_responses=[r.reasoning for r in results],
        vote_count=len(results),
        agree_count=all_agree_count,
    )


# --- LLM Call Builders ---

def _is_rate_limit(exc):
    """Detect rate-limit / 429 errors across providers."""
    cls_name = type(exc).__name__
    if cls_name in ('RateLimitError', 'ResourceExhausted', 'TooManyRequests'):
        return True
    status = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return any(p in msg for p in ('429', 'rate limit', 'resource_exhausted',
                                   'too many requests', 'quota exceeded'))


class _RateLimited(Exception):
    """Raised when a judge LLM call hits rate limits."""
    pass


def _build_gemini_call_fn():
    """Build a Gemini call function. Returns (call_fn, model_name) or (None, None)."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return None, None
    try:
        from google import genai as _genai
        client = _genai.Client(api_key=api_key)
        model = 'gemini-2.0-flash'

        def call_fn(prompt):
            from google.genai import types as gtypes
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=gtypes.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
            except Exception as e:
                if _is_rate_limit(e):
                    raise _RateLimited(str(e)) from e
                raise
            try:
                return resp.text.strip()
            except (TypeError, AttributeError):
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

        return call_fn, f'gemini/{model}'
    except Exception as e:
        logger.warning("Failed to initialize Gemini for judge: %s", e)
        return None, None


def _build_groq_call_fn():
    """Build a Groq call function. Returns (call_fn, model_name) or (None, None)."""
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        return None, None
    try:
        import groq as _groq
        client = _groq.Groq(api_key=api_key)
        model = 'llama-3.3-70b-versatile'

        def call_fn(prompt):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a classification quality judge for construction document control. Reply with ONLY valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=600,
                )
            except Exception as e:
                if _is_rate_limit(e):
                    raise _RateLimited(str(e)) from e
                raise
            return resp.choices[0].message.content.strip()

        return call_fn, f'groq/{model}'
    except Exception as e:
        logger.warning("Failed to initialize Groq for judge: %s", e)
        return None, None


def _build_claude_call_fn():
    """Build a Claude call function. Returns (call_fn, model_name) or (None, None)."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None, None
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=api_key)
        model = 'claude-sonnet-4-6'

        def call_fn(prompt):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=600,
                    system="You are a classification quality judge for construction document control. Reply with ONLY valid JSON.",
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as e:
                if _is_rate_limit(e):
                    raise _RateLimited(str(e)) from e
                raise
            return resp.content[0].text.strip()

        return call_fn, f'claude/{model}'
    except Exception as e:
        logger.warning("Failed to initialize Claude for judge: %s", e)
        return None, None


class ClassificationJudge:
    """Compound-scoring judge using majority voting with provider fallback."""

    # Cooldown (seconds) after a provider hits rate-limit
    _COOLDOWN = 120

    def __init__(self, providers, tracker, config_context, vote_count=3):
        """providers: list of (call_fn, model_name) tuples, tried in order."""
        self._providers = providers
        self._model_name = providers[0][1]  # primary model for DB storage
        self._tracker = tracker
        self._config_context = config_context
        self._vote_count = vote_count
        # {model_name: timestamp} — skip provider until cooldown expires
        self._cooldowns = {}

    def judge_single(self, email_data, classification):
        """Make one LLM call to judge a classification, with provider fallback."""
        prompt = _build_judge_prompt(email_data, classification, self._config_context)
        now = time.time()
        for call_fn, model_name in self._providers:
            # Skip providers in cooldown
            if model_name in self._cooldowns:
                if now < self._cooldowns[model_name]:
                    continue
                del self._cooldowns[model_name]
            try:
                raw_text = call_fn(prompt)
                result = _parse_judge_response(raw_text)
                self._model_name = model_name
                return result
            except _RateLimited:
                logger.warning("Judge rate-limited on %s, cooling down %ds",
                               model_name, self._COOLDOWN)
                self._cooldowns[model_name] = now + self._COOLDOWN
                continue
            except Exception as e:
                logger.warning("Judge call failed on %s: %s", model_name, e)
                return JudgeResult(
                    quality_score=0.5,
                    reasoning=f'Judge call failed: {e}',
                    error=str(e),
                )
        return JudgeResult(
            quality_score=0.5,
            reasoning='All judge providers rate-limited',
            error='all_rate_limited',
        )

    def judge(self, email_data, classification, pending_email_id, judge_type='post_scan'):
        """Compound judge: N calls + aggregate + store to DB.

        Returns None if all votes failed (does not store garbage results).
        """
        individual_results = []
        for i in range(self._vote_count):
            result = self.judge_single(email_data, classification)
            individual_results.append(result)
            # Rate limiting between votes
            if i < self._vote_count - 1:
                time.sleep(2)

        final = _aggregate_votes(individual_results)

        # Don't store if all votes failed
        if final.error:
            logger.warning("All judge votes failed for email #%d: %s", pending_email_id, final.reasoning)
            return None

        # Store to DB
        try:
            self._tracker.store_judge_result(
                pending_email_id=pending_email_id,
                message_id=email_data.get('message_id', ''),
                judge_model=self._model_name,
                primary_classifier=classification.get('classifier_method', ''),
                quality_score=final.quality_score,
                field_scores=final.field_scores,
                field_verdicts=final.field_verdicts,
                suggested_corrections=final.suggested_corrections,
                reasoning=final.reasoning,
                judge_type=judge_type,
                raw_responses=final.raw_responses,
                vote_count=final.vote_count,
                agree_count=final.agree_count,
            )
        except Exception as e:
            logger.error("Failed to store judge result: %s", e)

        return final

    def run_unjudged(self, limit=20, progress_callback=None):
        """Batch judge all unjudged pending emails."""
        unjudged_ids = self._tracker.get_unjudged_pending_ids(limit=limit)
        total = len(unjudged_ids)
        judged = 0
        errors = 0

        if progress_callback:
            progress_callback(0, total)

        for pid in unjudged_ids:
            email_data = self._tracker.get_pending_email(pid)
            if not email_data:
                errors += 1
                if progress_callback:
                    progress_callback(judged + errors, total)
                continue

            classification = {
                'doc_type': email_data.get('doc_type', ''),
                'discipline': email_data.get('discipline', ''),
                'department': email_data.get('department', ''),
                'response_required': bool(email_data.get('response_required', 0)),
                'classifier_method': email_data.get('classifier_method', ''),
            }

            try:
                result = self.judge(email_data, classification, pid, judge_type='post_scan')
                if result:
                    judged += 1
                else:
                    errors += 1
            except Exception as e:
                logger.error("Failed to judge email #%d: %s", pid, e)
                errors += 1

            if progress_callback:
                progress_callback(judged + errors, total)

        return {'judged': judged, 'errors': errors, 'total': total}


def _build_config_context(class_cfg, departments, contacts):
    """Build the full config context dict that the judge prompt needs."""
    type_keywords = class_cfg.get('type_keywords', {})
    discipline_keywords = class_cfg.get('discipline_keywords', {})

    return {
        'doc_types': list(type_keywords.keys()),
        'disciplines': list(discipline_keywords.keys()),
        'scope_keywords': class_cfg.get('scope_keywords', []),
        'response_phrases': class_cfg.get('response_required_phrases', []),
        'type_keywords': type_keywords,
        'discipline_keywords': discipline_keywords,
        'dept_text': _build_dept_context(departments),
        'contacts_text': _build_contacts_context(contacts),
    }


def create_judge(tracker, class_cfg, departments, contacts):
    """Factory: build a ClassificationJudge with all available providers.

    Args:
        tracker: ProcessingTracker instance
        class_cfg: classification config dict (from YAML)
        departments: list of department dicts from DB
        contacts: list of contact dicts from DB

    Collects all available providers (Gemini, Groq, Claude) for fallback.
    Returns None if no API keys available.
    """
    config_context = _build_config_context(class_cfg, departments, contacts)

    providers = []
    for builder in [_build_gemini_call_fn, _build_groq_call_fn, _build_claude_call_fn]:
        call_fn, model_name = builder()
        if call_fn:
            providers.append((call_fn, model_name))

    if not providers:
        logger.warning("No API keys available for judge — judge disabled")
        return None

    names = [m for _, m in providers]
    logger.info("Judge initialized with providers: %s", ', '.join(names))
    return ClassificationJudge(
        providers=providers,
        tracker=tracker,
        config_context=config_context,
    )
