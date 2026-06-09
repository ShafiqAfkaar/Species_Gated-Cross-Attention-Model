from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import get_settings
from .inference import VetDermInferenceService
from .knowledge import DiseaseCardStore
from .llm_providers import get_information_provider

app = FastAPI(
    title="VetDerm SGCA Triage API",
    version="0.1.0",
    description="Image triage and education API. Not a diagnostic or treatment system.",
)


@lru_cache(maxsize=1)
def service() -> VetDermInferenceService:
    svc = VetDermInferenceService(get_settings())
    svc.load()
    return svc


@lru_cache(maxsize=1)
def cards() -> DiseaseCardStore:
    return DiseaseCardStore(get_settings().disease_cards_path)


@lru_cache(maxsize=1)
def information_provider():
    return get_information_provider()


@app.get("/health")
def health() -> dict[str, object]:
    svc = service()
    return {
        "ok": True,
        "checkpoint": str(svc.settings.checkpoint_path),
        "device": str(svc.device),
        "species": svc.species_names,
        "diseases": len(svc.disease_names),
        "calibration_source": svc.calibration.source,
    }


@app.post("/predict")
async def predict(
    species: Literal["Cat", "Cattles", "Dog"] = Form(...),
    image: UploadFile = File(...),
    include_gradcam: bool = Form(True),
) -> dict[str, object]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image file.")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    try:
        result = service().predict_bytes(image_bytes, user_species=species, include_gradcam=include_gradcam)
        if result.get("status") == "unsupported_image":
            result["information"] = {
                "shown": False,
                "reason": result["referral"]["banner"],
                "safety_note": "Unsupported image. No disease information was generated.",
            }
            return result
        top1 = result["top3_diseases"][0]["disease"]
        card = cards().get(top1)
        result["information"] = information_provider().compose(
            card,
            suppress=bool(result["referral"]["suppress_information_panel"]),
            referral_banner=str(result["referral"]["banner"]),
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
