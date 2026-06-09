from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from vetderm_agent.config import get_settings
from vetderm_agent.inference import VetDermInferenceService
from vetderm_agent.knowledge import DiseaseCardStore
from vetderm_agent.llm_providers import get_information_provider


@st.cache_resource
def get_service() -> VetDermInferenceService:
    svc = VetDermInferenceService(get_settings())
    svc.load()
    return svc


@st.cache_resource
def get_cards() -> DiseaseCardStore:
    return DiseaseCardStore(get_settings().disease_cards_path)


@st.cache_resource
def get_info_provider():
    return get_information_provider()


st.set_page_config(page_title="VetDerm SGCA Triage", layout="wide")
st.title("VetDerm SGCA Triage")
st.caption("Decision-support and education only. This is not a veterinary diagnosis or treatment plan.")

species = st.selectbox("Species", ["Cat", "Cattles", "Dog"], index=0)
uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png", "webp", "bmp"])
include_gradcam = st.checkbox("Generate Grad-CAM overlay", value=True)

if uploaded:
    image_bytes = uploaded.getvalue()
    left, right = st.columns([1, 1])
    with left:
        st.image(image_bytes, caption="Uploaded image", use_container_width=True)

    if st.button("Run SGCA triage", type="primary"):
        with st.spinner("Running SGCA Unified with MC-dropout uncertainty..."):
            result = get_service().predict_bytes(
                image_bytes,
                user_species=species,
                include_gradcam=include_gradcam,
            )
        with right:
            st.subheader("Referral")
            st.warning(result["referral"]["banner"])
            if result["referral"]["signals"]:
                st.write("Signals:", ", ".join(result["referral"]["signals"]))

            if result.get("status") == "unsupported_image":
                st.subheader("Domain gate")
                st.json(result.get("domain_gate", {}))
                st.error("No disease information was generated for this unsupported image.")
                st.stop()

            st.subheader("Top visual matches")
            st.dataframe(
                pd.DataFrame(result["top3_diseases"])[["rank", "disease", "probability"]],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Possible alternatives")
            st.dataframe(
                pd.DataFrame(result["conformal_prediction_set"])[["disease", "probability"]],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Uncertainty")
            st.json(result["uncertainty"])

        artifacts = result.get("gradcam_artifacts") or {}
        overlay = artifacts.get("overlay")
        if overlay and Path(overlay).exists():
            st.subheader("Grad-CAM overlay")
            st.image(overlay, use_container_width=True)
        elif artifacts.get("error"):
            st.info(f"Grad-CAM unavailable: {artifacts['error']}")

        top1 = result["top3_diseases"][0]["disease"]
        info = get_info_provider().compose(
            get_cards().get(top1),
            suppress=bool(result["referral"]["suppress_information_panel"]),
            referral_banner=str(result["referral"]["banner"]),
        )
        st.subheader("Educational information")
        if not info["shown"]:
            st.error(info["reason"])
        elif "text" in info:
            st.write(info["text"])
            st.caption(info["safety_note"])
        else:
            st.write(info["overview"])
            st.write("Typical signs:", ", ".join(info["typical_signs"]))
            st.write("Transmission:", info["transmission"])
            st.write("Ask your vet:", ", ".join(info["ask_your_vet"]))
            st.write("Urgent flags:", ", ".join(info["urgent_flags"]))
            st.caption(info["safety_note"])
