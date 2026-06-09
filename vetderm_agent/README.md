# VetDerm SGCA Triage Agent

This is a paper-companion scaffold for a veterinary skin-image triage assistant.
It is intentionally built as decision support and education, not diagnosis or treatment.

## Architecture

1. **Inference service**
   - Loads `unified_sgca_best.pt` once.
   - Requires a user-selected species: `Cat`, `Cattles` or `Dog`.
   - Runs MC-dropout with `T=30` by default.
   - Applies species-conditioned decoding.
   - Returns top-3 predictions, predictive entropy, mutual information, an APS conformal prediction set, referral signals and an optional Grad-CAM overlay.

2. **Information layer**
   - Default: local curated disease cards in `knowledge/disease_cards.json`.
   - Optional: Google AI Studio / Gemini provider by setting `VETDERM_INFO_PROVIDER=gemini`.
   - Optional: local Ollama provider by setting `VETDERM_INFO_PROVIDER=ollama`.
   - The LLM is allowed only to rewrite the curated card. It is not allowed to add treatment, dosing, drug names or diagnosis.

3. **Referral logic**
   - Recommends or requires veterinary review when any uncertainty or safety signal trips:
     - any non-normal triage label,
     - predictive entropy above the validation-calibrated 90th percentile,
     - conformal set size greater than 1 when top-1 probability is below `0.95`,
     - top-1 probability below `0.60`,
     - possible zoonotic/contagious label,
     - possible herd-health or reportable livestock label,
     - poor image quality warning.
   - Two or more major signals, or a herd-health/reportable livestock signal, suppress the information panel until veterinary review is acknowledged in a future UI step.
   - This differs from a naive "set size > 1" rule because APS sets can be large even for very high-confidence predictions when the validation probabilities are extremely peaked.

## Install

```bash
./.venv-wsl/bin/pip install fastapi 'uvicorn[standard]' python-multipart streamlit requests
```

## Run the FastAPI service

```bash
./.venv-wsl/bin/python -m uvicorn vetderm_agent.api:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction example:

```bash
curl -X POST http://localhost:8000/predict \
  -F "species=Cat" \
  -F "include_gradcam=true" \
  -F "image=@/path/to/image.jpg"
```

## Run the Streamlit demo

```bash
./.venv-wsl/bin/streamlit run vetderm_agent/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

## Optional local LLM

Install and run Ollama, then set:

```bash
export VETDERM_INFO_PROVIDER=ollama
export VETDERM_OLLAMA_MODEL=gemma3
```

The app will call `http://localhost:11434/api/chat`. If Ollama is unavailable, it falls back to the static disease-card output.

## Optional Gemini API information layer

Set the Google AI Studio key only in the backend environment:

```bash
export VETDERM_INFO_PROVIDER=gemini
export GEMINI_API_KEY=your_google_ai_studio_key
export VETDERM_GEMINI_MODEL=gemini-2.5-flash
```

The Gemini provider sends only sanitized text: the predicted disease label, referral banner and curated disease-card content. It does not send the uploaded image, filename or owner identifiers. Successful Gemini rewrites are cached at `vetderm_agent/cache/information_cache.json` by model, disease label, referral level and card hash. If the key is missing, the request fails or the response contains blocked treatment/medication language, the app falls back to the static disease-card output.

Smoke-test the three built-in Cat/Cattles/Dog samples:

```bash
./.venv-wsl/bin/python -m vetderm_agent.smoke_test_agent --provider gemini --mc-passes 5
```

Run one image from every disease folder. The command asks for the Gemini key without echoing it:

```bash
./.venv-wsl/bin/python -m vetderm_agent.check_agent
```

## Important safety boundary

The app must always display:

- "Decision-support and education only. Not a diagnosis."
- A permanent "See a veterinarian" pathway.
- The conformal prediction set as possible visual alternatives.
- The Grad-CAM overlay when available.

Do not add treatment, drug names, dosing, home remedies or prognosis text to the disease cards or prompts.
