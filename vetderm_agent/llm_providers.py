from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from .knowledge import DiseaseCard, compose_safe_information


SAFETY_SYSTEM_PROMPT = """You are a veterinary education assistant for an image-triage tool.
You must not diagnose, prescribe, recommend treatment, name drugs, give doses, give home remedies,
or imply that the image model is clinically definitive. Use only the provided disease card.
Allowed content: disease overview, typical visual signs, transmission/zoonotic caution, what to ask a veterinarian,
and urgent warning signs. If the referral decision says veterinary review is required, say that first."""


INFORMATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overview": {
            "type": "string",
            "description": "Two or three plain-language sentences based only on the provided card.",
        },
        "typical_signs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short bullets copied or paraphrased from the provided typical_signs only.",
        },
        "transmission": {
            "type": "string",
            "description": "Plain-language transmission or zoonotic caution from the provided card only.",
        },
        "ask_your_vet": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Questions to ask a veterinarian, based only on the provided card.",
        },
        "urgent_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Warning signs from the provided card only.",
        },
        "safety_note": {
            "type": "string",
            "description": "Must say this is educational only and not diagnosis, treatment, medication, dosing or prognosis.",
        },
    },
    "required": [
        "overview",
        "typical_signs",
        "transmission",
        "ask_your_vet",
        "urgent_flags",
        "safety_note",
    ],
}

FORBIDDEN_OUTPUT_RE = re.compile(
    r"\b("
    r"mg|ml|dose|dosage|administer|apply|prescribe|prescription|"
    r"antibiotic|antifungal cream|antifungal medication|steroid|"
    r"ivermectin|permethrin|ketoconazole|miconazole|clotrimazole|"
    r"terbinafine|griseofulvin|chlorhexidine|prednisone|"
    r"home remedy|coconut oil|tea tree oil|apple cider vinegar"
    r")\b",
    re.IGNORECASE,
)


DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "information_cache.json"


class InformationProvider(Protocol):
    def compose(self, card: DiseaseCard, *, suppress: bool, referral_banner: str | None = None) -> dict[str, Any]:
        ...


def _information_prompt(card: DiseaseCard, referral_banner: str | None) -> dict[str, Any]:
    return {
        "referral_banner": referral_banner,
        "disease_card": asdict(card),
        "rules": [
            "Use only the disease_card facts.",
            "Do not add clinical facts that are not in the card.",
            "Do not provide diagnosis, treatment, medicines, dosing, home remedies or prognosis.",
            "Keep the language suitable for a pet or livestock owner.",
        ],
    }


def _fallback(card: DiseaseCard, *, suppress: bool, error: str | None = None) -> dict[str, Any]:
    fallback = compose_safe_information(card, suppress=suppress)
    if error and fallback.get("shown", False):
        fallback["provider"] = "template_fallback"
        fallback["llm_error"] = error
    return fallback


def _referral_level_from_banner(referral_banner: str | None) -> str:
    banner = (referral_banner or "").lower()
    if "visit a veterinarian" in banner:
        return "vet_required"
    if "recommended" in banner:
        return "vet_recommended"
    return "educational_only"


def _cache_key(card: DiseaseCard, *, model: str, referral_banner: str | None) -> str:
    card_payload = json.dumps(asdict(card), sort_keys=True, ensure_ascii=False)
    card_hash = sha256(card_payload.encode("utf-8")).hexdigest()[:16]
    referral_level = _referral_level_from_banner(referral_banner)
    return f"gemini|{model}|{card.disease}|{referral_level}|{card_hash}"


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_cached_info(path: Path, key: str) -> dict[str, Any] | None:
    payload = _load_cache(path).get(key)
    if not isinstance(payload, dict):
        return None
    cached = dict(payload)
    cached["provider"] = "gemini_cache"
    cached.pop("llm_error", None)
    return cached


def _write_cached_info(path: Path, key: str, payload: dict[str, Any]) -> None:
    try:
        cache = _load_cache(path)
        cache[key] = payload
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return


def _extract_gemini_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "\n".join(str(part.get("text", "")) for part in parts).strip()


def _loads_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    loaded = json.loads(cleaned)
    if not isinstance(loaded, dict):
        raise ValueError("LLM output was not a JSON object.")
    return loaded


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _render_information_payload(
    payload: dict[str, Any], card: DiseaseCard, *, provider: str, model: str
) -> dict[str, Any]:
    typical_signs = _coerce_string_list(payload.get("typical_signs"))
    ask_your_vet = _coerce_string_list(payload.get("ask_your_vet"))
    urgent_flags = _coerce_string_list(payload.get("urgent_flags"))
    rendered = {
        "shown": True,
        "provider": provider,
        "model": model,
        "disease": card.disease,
        "overview": str(payload.get("overview", "")).strip(),
        "typical_signs": typical_signs,
        "transmission": str(payload.get("transmission", "")).strip(),
        "ask_your_vet": ask_your_vet,
        "urgent_flags": urgent_flags,
        "safety_note": (
            str(payload.get("safety_note", "")).strip()
            or "Educational information only. This tool does not provide treatment, medication, dosing, prognosis or a diagnosis."
        ),
        "source_card": asdict(card),
    }
    text = "\n\n".join(
        part
        for part in [
            f"{card.disease}",
            rendered["overview"],
            "Typical signs: " + "; ".join(typical_signs) if typical_signs else "",
            "Transmission: " + rendered["transmission"] if rendered["transmission"] else "",
            "Ask your vet: " + "; ".join(ask_your_vet) if ask_your_vet else "",
            "Urgent flags: " + "; ".join(urgent_flags) if urgent_flags else "",
            rendered["safety_note"],
        ]
        if part
    )
    if FORBIDDEN_OUTPUT_RE.search(text):
        raise ValueError("LLM output contained blocked treatment or medication language.")
    rendered["text"] = text
    return rendered


class TemplateInformationProvider:
    def compose(self, card: DiseaseCard, *, suppress: bool, referral_banner: str | None = None) -> dict[str, Any]:
        return compose_safe_information(card, suppress=suppress)


class OllamaInformationProvider:
    """Optional local LLM provider.

    Requires Ollama to be running locally. This is intentionally limited to
    rewriting a curated card; it is not allowed to add new veterinary facts.
    """

    def __init__(self, *, model: str | None = None, base_url: str | None = None, timeout: int = 30):
        self.model = model or os.environ.get("VETDERM_OLLAMA_MODEL", "gemma3")
        self.base_url = (base_url or os.environ.get("VETDERM_OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def compose(self, card: DiseaseCard, *, suppress: bool, referral_banner: str | None = None) -> dict[str, Any]:
        if suppress:
            return compose_safe_information(card, suppress=True)
        prompt = _information_prompt(card, referral_banner)
        body = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return {
                "shown": True,
                "provider": "ollama",
                "model": self.model,
                "text": data.get("message", {}).get("content", "").strip(),
                "source_card": asdict(card),
                "safety_note": (
                    "Educational information only. This tool does not provide treatment, medication, dosing, prognosis or a diagnosis."
                ),
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return _fallback(card, suppress=False, error=str(exc))


class GeminiInformationProvider:
    """Google AI Studio / Gemini API provider.

    Only sanitized disease-card text is sent to the model. Uploaded
    images and user identifiers stay out of the LLM request.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_path: Path | None = None,
        timeout: int = 30,
    ):
        self.model = model or os.environ.get("VETDERM_GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.base_url = (
            base_url or os.environ.get("VETDERM_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        ).rstrip("/")
        self.cache_path = cache_path or Path(os.environ.get("VETDERM_LLM_CACHE", DEFAULT_CACHE_PATH))
        self.cache_enabled = os.environ.get("VETDERM_DISABLE_LLM_CACHE", "0").lower() not in {"1", "true", "yes"}
        self.timeout = timeout

    def compose(self, card: DiseaseCard, *, suppress: bool, referral_banner: str | None = None) -> dict[str, Any]:
        if suppress:
            return compose_safe_information(card, suppress=True)
        cache_key = _cache_key(card, model=self.model, referral_banner=referral_banner)
        if self.cache_enabled:
            cached = _read_cached_info(self.cache_path, cache_key)
            if cached is not None:
                return cached
        if not self.api_key:
            return _fallback(
                card,
                suppress=False,
                error="Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable.",
            )

        body = {
            "system_instruction": {"parts": [{"text": SAFETY_SYSTEM_PROMPT}]},
            "contents": [
                {
                    "parts": [
                        {
                            "text": json.dumps(
                                _information_prompt(card, referral_banner),
                                ensure_ascii=False,
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
                "maxOutputTokens": 700,
                "responseMimeType": "application/json",
                "responseJsonSchema": INFORMATION_OUTPUT_SCHEMA,
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/models/{self.model}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            payload = _loads_json_text(_extract_gemini_text(data))
            rendered = _render_information_payload(payload, card, provider="gemini", model=self.model)
            if self.cache_enabled:
                _write_cached_info(self.cache_path, cache_key, rendered)
            return rendered
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            return _fallback(card, suppress=False, error=str(exc))


def get_information_provider() -> InformationProvider:
    provider = os.environ.get("VETDERM_INFO_PROVIDER", "template").lower()
    if provider == "gemini":
        return GeminiInformationProvider()
    if provider == "ollama":
        return OllamaInformationProvider()
    return TemplateInformationProvider()
