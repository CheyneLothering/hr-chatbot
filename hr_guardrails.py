import json
from typing import Tuple

from openai import OpenAI

HR_KEYWORDS = [
    "leave", "vacation", "benefits", "pay",
    "termination", "harassment", "policy",
    "performance", "manager", "employee"
]

CLASSIFIER_SYSTEM_PROMPT = """
You are an HR domain classifier for a workplace assistant.

Return a JSON object with:
- hr_related: true or false
- reason: short, plain-English explanation

HR topics include: benefits, leave, payroll, performance, policies, harassment,
training, onboarding, employee relations, and workplace issues.
Non-HR topics include: coding, math, cooking, travel, jokes, and trivia.
If unsure, return hr_related: false.
"""


def _keyword_fallback(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in HR_KEYWORDS)


def is_hr_related(text: str, client: OpenAI) -> Tuple[bool, str]:
    if not text or not text.strip():
        return False, "empty"

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        hr_related = bool(data.get("hr_related"))
        reason = str(data.get("reason", "")).strip()
        return hr_related, reason
    except Exception:
        return _keyword_fallback(text), "fallback"