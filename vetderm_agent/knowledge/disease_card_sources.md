# Local Disease Card Source Notes

The local cards in `disease_cards.json` are written as educational triage text for the SGCA agent. They are intentionally conservative: no diagnosis, treatment plan, medication name, dose, home remedy, or prognosis is generated from the image label.

Primary source families used for the May 2026 card refresh:

- Merck Veterinary Manual, dermatophytosis in dogs/cats and cattle: ringworm signs, zoonotic risk, and diagnostic framing.
- Merck Veterinary Manual, mange in dogs/cats and cattle: mite-associated pruritus, alopecia, crusting, contagion and sampling framing.
- Merck Veterinary Manual, mite infestations and outer-ear disorders: ear mite and pinna irritation framing.
- Merck Veterinary Manual, conjunctiva and cornea disorders: red eye, discharge, squinting, corneal-risk and veterinary-exam framing.
- Merck Veterinary Manual, atopic dermatitis, feline atopic dermatitis, pruritus, pyoderma and acute moist dermatitis: allergy, dermatitis, secondary infection and hot-spot framing.
- Merck Veterinary Manual, ticks of dogs: tick attachment, tick-borne disease risk and systemic warning signs.
- Merck Veterinary Manual, lumpy skin disease in cattle, and WOAH lumpy skin disease page: cattle nodules, transmission, herd-health and official-guidance framing.
- WOAH foot-and-mouth disease page: vesicular lesions, high contagiousness, transboundary disease status, and official reporting urgency.

Operational rule: if an LLM provider is unavailable, rate-limited or disabled, the agent can answer from these cards alone through `TemplateInformationProvider`.
