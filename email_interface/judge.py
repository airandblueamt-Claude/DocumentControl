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

JUDGE_FIELDS = ['doc_type', 'discipline', 'department', 'response_required']


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


def _build_judge_prompt(email_data, classification, departments):
    """Build structured prompt asking the LLM to evaluate classification quality."""
    dept_list = ', '.join(departments) if departments else 'none configured'

    prompt = (
        "You are a quality assurance judge for a document control classification system.\n"
        "An AI classifier processed an incoming email and assigned the fields below.\n"
        "Your job: evaluate whether each field was classified correctly.\n\n"
        "--- EMAIL ---\n"
        f"From: {email_data.get('sender_name', '')} <{email_data.get('sender', '')}>\n"
        f"Subject: {email_data.get('subject', '')}\n"
        f"Body (excerpt): {(email_data.get('body', '') or '')[:1500]}\n"
        f"Attachment count: {email_data.get('attachment_count', 0)}\n\n"
        "--- CLASSIFICATION TO EVALUATE ---\n"
        f"doc_type: {classification.get('doc_type', '')}\n"
        f"discipline: {classification.get('discipline', '')}\n"
        f"department: {classification.get('department', '')}\n"
        f"response_required: {classification.get('response_required', False)}\n"
        f"Classifier used: {classification.get('classifier_method', 'unknown')}\n\n"
        "--- CONTEXT ---\n"
        f"Available departments: {dept_list}\n\n"
        "Evaluate each field. Reply with ONLY valid JSON:\n"
        "{\n"
        '  "quality_score": 0.0 to 1.0 (overall classification quality),\n'
        '  "field_scores": {"doc_type": 0.0-1.0, "discipline": 0.0-1.0, '
        '"department": 0.0-1.0, "response_required": 0.0-1.0},\n'
        '  "field_verdicts": {"doc_type": "agree" or "disagree", '
        '"discipline": "agree" or "disagree", '
        '"department": "agree" or "disagree", '
        '"response_required": "agree" or "disagree"},\n'
        '  "suggested_corrections": {"field_name": "suggested value"} (only for fields you disagree with, omit if all correct),\n'
        '  "reasoning": "brief explanation of your assessment"\n'
        "}"
    )
    return prompt


def _parse_judge_response(text):
    """Parse a single judge LLM response into a JudgeResult."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        lines = lines[1:]  # remove opening fence
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

    # Merge suggested corrections (only from votes that disagreed)
    merged_corrections = {}
    for r in valid:
        for fld, val in r.suggested_corrections.items():
            if fld not in merged_corrections:
                merged_corrections[fld] = []
            merged_corrections[fld].append(val)
    # Pick most common suggestion per field
    final_corrections = {}
    for fld, suggestions in merged_corrections.items():
        if final_verdicts.get(fld) == 'disagree':
            counter = Counter(suggestions)
            final_corrections[fld] = counter.most_common(1)[0][0]

    # Count fields where all voters agree
    all_agree_count = sum(
        1 for f in JUDGE_FIELDS
        if all(r.field_verdicts.get(f, 'agree') == 'agree' for r in valid)
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

def _build_gemini_call_fn():
    """Build a Gemini call function. Returns None if no API key."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return None, None
    try:
        from google import genai as _genai
        client = _genai.Client(api_key=api_key)
        model = 'gemini-2.0-flash-lite'

        def call_fn(prompt):
            from google.genai import types as gtypes
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=gtypes.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            # Extract text from response
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
    """Build a Groq call function. Returns None if no API key."""
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        return None, None
    try:
        import groq as _groq
        client = _groq.Groq(api_key=api_key)
        model = 'llama-3.3-70b-versatile'

        def call_fn(prompt):
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a classification quality judge. Reply with ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
            )
            return resp.choices[0].message.content.strip()

        return call_fn, f'groq/{model}'
    except Exception as e:
        logger.warning("Failed to initialize Groq for judge: %s", e)
        return None, None


def _build_claude_call_fn():
    """Build a Claude call function. Returns None if no API key."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None, None
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=api_key)
        model = 'claude-sonnet-4-5-20250929'

        def call_fn(prompt):
            resp = client.messages.create(
                model=model,
                max_tokens=500,
                system="You are a classification quality judge. Reply with ONLY valid JSON.",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()

        return call_fn, f'claude/{model}'
    except Exception as e:
        logger.warning("Failed to initialize Claude for judge: %s", e)
        return None, None


class ClassificationJudge:
    """Compound-scoring judge using majority voting."""

    def __init__(self, call_fn, model_name, tracker, departments, contacts, vote_count=3):
        self._call_fn = call_fn
        self._model_name = model_name
        self._tracker = tracker
        self._departments = [d['name'] for d in departments] if departments else []
        self._contacts = contacts or []
        self._vote_count = vote_count

    def judge_single(self, email_data, classification):
        """Make one LLM call to judge a classification."""
        prompt = _build_judge_prompt(email_data, classification, self._departments)
        try:
            raw_text = self._call_fn(prompt)
            result = _parse_judge_response(raw_text)
            return result
        except Exception as e:
            logger.warning("Judge single call failed: %s", e)
            return JudgeResult(
                quality_score=0.5,
                reasoning=f'Judge call failed: {e}',
                error=str(e),
            )

    def judge(self, email_data, classification, pending_email_id, judge_type='post_scan'):
        """Compound judge: N calls + aggregate + store to DB."""
        individual_results = []
        for i in range(self._vote_count):
            result = self.judge_single(email_data, classification)
            individual_results.append(result)
            # Rate limiting between votes
            if i < self._vote_count - 1:
                time.sleep(2)

        final = _aggregate_votes(individual_results)

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
                continue

            classification = {
                'doc_type': email_data.get('doc_type', ''),
                'discipline': email_data.get('discipline', ''),
                'department': email_data.get('department', ''),
                'response_required': bool(email_data.get('response_required', 0)),
                'classifier_method': email_data.get('classifier_method', ''),
            }

            try:
                self.judge(email_data, classification, pid, judge_type='post_scan')
                judged += 1
            except Exception as e:
                logger.error("Failed to judge email #%d: %s", pid, e)
                errors += 1

            if progress_callback:
                progress_callback(judged + errors, total)

        return {'judged': judged, 'errors': errors, 'total': total}


def create_judge(tracker, departments, contacts):
    """Factory: build a ClassificationJudge using the best available API.

    Tries Gemini -> Groq -> Claude. Returns None if no API keys available.
    """
    for builder in [_build_gemini_call_fn, _build_groq_call_fn, _build_claude_call_fn]:
        call_fn, model_name = builder()
        if call_fn:
            logger.info("Judge initialized with %s", model_name)
            return ClassificationJudge(
                call_fn=call_fn,
                model_name=model_name,
                tracker=tracker,
                departments=departments,
                contacts=contacts,
            )

    logger.warning("No API keys available for judge — judge disabled")
    return None
