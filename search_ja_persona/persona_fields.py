PERSONA_TEXT_FIELDS: tuple[str, ...] = (
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "persona",
)

DEFAULT_PERSONA_FIELDS = PERSONA_TEXT_FIELDS

# Nemotron-Personas-Finance-Japan: three finance personas plus the six above.
FINANCE_PERSONA_TEXT_FIELDS: tuple[str, ...] = (
    "finance_persona",
    "financial_behavior_persona",
    "financial_concerns_persona",
    *PERSONA_TEXT_FIELDS,
)

__all__ = [
    "DEFAULT_PERSONA_FIELDS",
    "FINANCE_PERSONA_TEXT_FIELDS",
    "PERSONA_TEXT_FIELDS",
]
