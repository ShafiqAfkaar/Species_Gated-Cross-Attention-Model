# Strategy Review

Claude's three-layer strategy is directionally good, but the safer implementation is stricter:

1. **Make species required.** Do not let the model infer deployment species as the only source of species metadata. The paper already shows that known-species decoding is a major external-domain gain.

2. **Do not let the LLM reason medically.** The LLM should only rewrite curated disease cards retrieved by exact disease label. The image model and deterministic safety rules decide what is shown.

3. **Use Unified SGCA first.** SGCA Cross-Attention has slightly better external macro-F1 and PR-AUC, but Unified SGCA exposes MC-dropout uncertainty, conformal sets and Grad-CAM from one checkpoint. That makes it better for a triage tool.

4. **Use Cross-Attention later as a disagreement check.** A second-model agreement signal is useful, but not needed for the first scaffold.

5. **Use APS conformal sets, not simple probability-threshold conformal sets.** The current validation probabilities are highly peaked, so a `p >= threshold` conformal set degenerates to singletons. APS gives a more useful alternatives list, but set size alone should not be a hard uncertainty trigger. In the scaffold, the conformal signal trips only when the set has more than one label and top-1 probability is below 0.95.

6. **Prefer local information by default.** Cloud LLMs are useful for research demos, but privacy and rate limits are unstable. Local curated cards plus optional Ollama is the better first implementation. Gemini/Groq can be added behind the same provider interface later.

7. **Escalate conservatively.** Ringworm, scabies and mange-like labels always trigger a veterinary-review signal because of possible transmission or zoonotic concern.

## Current scaffold status

- `vetderm_agent/inference.py`: model loading, MC-dropout, species-conditioned top-3, APS conformal set, referral rules, Grad-CAM artifact export and audit logging.
- `vetderm_agent/api.py`: FastAPI upload endpoint.
- `vetderm_agent/streamlit_app.py`: local demo UI.
- `vetderm_agent/knowledge/disease_cards.json`: safe educational disease cards for all 21 labels.
- `vetderm_agent/llm_providers.py`: default template provider and optional Ollama provider.
