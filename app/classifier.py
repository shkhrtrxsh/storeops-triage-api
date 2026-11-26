import json
import os
import re
from typing import Protocol

import httpx
from pydantic import ValidationError

from app.schemas import ClassificationResult, IncidentCategory, IncidentCreate, IncidentPriority


class ClassifierError(Exception):
    """Raised when the upstream classifier returns malformed or invalid output."""


class ClassifierTimeoutError(ClassifierError):
    """Raised when the upstream classifier times out."""


class IncidentClassifier(Protocol):
    async def classify(self, incident: IncidentCreate) -> ClassificationResult:
        ...


SYSTEM_PROMPT = """You are a retail store operations incident triage classifier.
Classify the incident into ONE allowed category and ONE allowed priority, or route to human review if information is insufficient, vague, or conflicting.

Allowed categories:
- pos_billing: POS terminal crashes, UPI/card payment failures, billing/checkout errors
- kitchen_display: KDS screen freezes, order routing delays, kitchen ticket sync issues
- network_connectivity: Store Wi-Fi/ISP outage, VPN drop, offline store server
- inventory_sync: Stock count mismatch, SKU sync failure, wastage/GRN sync errors
- hardware_peripheral: Thermal receipt printer, barcode scanner, cash drawer, UPS hardware faults

Allowed priorities:
- P1: Store-wide outage, all billing/POS down, complete network outage during service
- P2: Major degradation affecting multiple stations or active order flow with workaround
- P3: Single peripheral/station issue or non-blocking sync delay
- P4: Minor cosmetic issue or informational request

Rules:
1. If the report is vague, lacks actionable symptoms, or mixes conflicting unrelated failures without clear root impact, set needs_review=true and leave suggested_category=null and suggested_priority=null.
2. Return ONLY valid JSON matching:
{"suggested_category": string|null, "suggested_priority": string|null, "needs_review": boolean, "confidence": number, "reason": string}
"""


class RuleBasedClassifier:
    """Deterministic local classifier for offline execution, tests, and baseline evaluation."""

    CATEGORY_KEYWORDS: dict[IncidentCategory, tuple[str, ...]] = {
        IncidentCategory.POS_BILLING: ("pos", "billing", "upi", "payment", "checkout", "card terminal", "invoice"),
        IncidentCategory.KITCHEN_DISPLAY: ("kds", "kitchen display", "makeline", "kot", "order screen", "ticket routing"),
        IncidentCategory.NETWORK_CONNECTIVITY: ("internet", "isp", "wifi", "wi-fi", "vpn", "network", "offline", "packet loss"),
        IncidentCategory.INVENTORY_SYNC: ("inventory", "stock", "sku", "grn", "wastage", "catalog sync"),
        IncidentCategory.HARDWARE_PERIPHERAL: ("printer", "scanner", "cash drawer", "ups", "cable", "thermal paper"),
    }

    AMBIGUOUS_MARKERS = (
        "something is wrong",
        "not working properly",
        "please check",
        "maybe",
        "not sure",
        "weird issue",
        "help asap",
    )

    async def classify(self, incident: IncidentCreate) -> ClassificationResult:
        text = f"{incident.title} {incident.description}".lower()

        matched_categories = [
            cat
            for cat, keywords in self.CATEGORY_KEYWORDS.items()
            if any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in keywords)
        ]

        is_too_short = len(incident.description.split()) < 5
        has_ambiguous_phrase = any(marker in text for marker in self.AMBIGUOUS_MARKERS) and len(matched_categories) == 0
        has_conflict = len(matched_categories) > 1

        if is_too_short or has_ambiguous_phrase or has_conflict or not matched_categories:
            reason = (
                "Conflicting signals across multiple subsystems; routed for human triage."
                if has_conflict
                else "Insufficient or ambiguous symptom details; routed for human review."
            )
            return ClassificationResult(
                suggested_category=None,
                suggested_priority=None,
                needs_review=True,
                confidence=0.35 if has_conflict else 0.25,
                reason=reason,
            )

        category = matched_categories[0]

        if any(w in text for w in ("all ", "entire store", "completely down", "stopped accepting", "outage", "cannot bill")):
            priority = IncidentPriority.P1_CRITICAL
            confidence = 0.92
        elif any(w in text for w in ("slow", "intermittent", "delayed", "two counters", "degraded", "timeout")):
            priority = IncidentPriority.P2_HIGH
            confidence = 0.86
        elif any(w in text for w in ("single", "one ", "backup", "minor", "jammed")):
            priority = IncidentPriority.P3_MEDIUM
            confidence = 0.84
        else:
            priority = IncidentPriority.P3_MEDIUM
            confidence = 0.78

        return ClassificationResult(
            suggested_category=category,
            suggested_priority=priority,
            needs_review=False,
            confidence=confidence,
            reason=f"Matched {category.value} symptoms with {priority.value} operational impact.",
        )


class OpenAICompatibleClassifier:
    """Async LLM classifier using any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout_s: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    async def classify(self, incident: IncidentCreate) -> ClassificationResult:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Store: {incident.store_id}\nTitle: {incident.title}\nDescription: {incident.description}",
                },
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ClassifierTimeoutError("Upstream LLM classifier timed out") from exc
        except httpx.HTTPError as exc:
            raise ClassifierError(f"Upstream LLM request failed: {exc}") from exc

        try:
            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw_content)
            result = ClassificationResult.model_validate(parsed)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise ClassifierError(f"Malformed classifier output: {exc}") from exc

        if result.confidence < 0.65 or result.suggested_category is None or result.suggested_priority is None:
            return ClassificationResult(
                suggested_category=None,
                suggested_priority=None,
                needs_review=True,
                confidence=result.confidence,
                reason=result.reason,
            )
        return result


def build_default_classifier() -> IncidentClassifier:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if api_key:
        return OpenAICompatibleClassifier(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            timeout_s=float(os.getenv("LLM_TIMEOUT_S", "10.0")),
        )
    return RuleBasedClassifier()
