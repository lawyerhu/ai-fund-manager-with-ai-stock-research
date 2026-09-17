"""Fixed research-model settings for the AI stock-research skill.

Every new research request made by this skill uses the project's CCSwitch
provider with ``gpt-5.6-sol`` and ``medium`` reasoning effort. Luna screening
settings remain project-owned. The old interactive-session handoff utilities
are retained only to read historical artifacts; they are not the default
research execution path because they cannot verify a model's reasoning effort.
"""
from __future__ import annotations

import os

SESSION_PROVIDER = "CURRENT_CONVERSATION"
SESSION_MODEL_PROVENANCE = "SESSION_DECLARED_NOT_API_VERIFIED"
SESSION_MODEL_ENV = "AI_RESEARCH_SESSION_MODEL"
# Placeholder only. The real name is declared per run by the active session
# (`--session-model`), so no specific model is hardcoded in this skill.
DEFAULT_SESSION_MODEL = "CURRENT_CONVERSATION_MODEL"
SESSION_EFFORT_STATUS = "NOT_EXPOSED_BY_SESSION"

# These are intentionally fixed rather than inheriting an arbitrary gateway
# default. A stale environment override must fail closed instead of silently
# routing a new research run to another model.
RESEARCH_MODEL = "gpt-5.6-sol"
RESEARCH_EFFORT = "medium"
API_MODEL_ENV = "AI_RESEARCH_API_MODEL"
API_EFFORT_ENV = "AI_RESEARCH_API_EFFORT"


def session_model(declared=None):
    """Session-declared research model; never verified against an API."""
    value = (declared or os.environ.get(SESSION_MODEL_ENV) or "").strip()
    return value or DEFAULT_SESSION_MODEL


def api_model():
    """Return the fixed model used for all new skill research requests."""
    requested = (os.environ.get(API_MODEL_ENV) or "").strip()
    if requested and requested != RESEARCH_MODEL:
        raise ValueError(
            f"{API_MODEL_ENV} must be {RESEARCH_MODEL} for this skill; got {requested}"
        )
    return RESEARCH_MODEL


def api_effort():
    """Return the fixed reasoning effort used for all new skill research requests."""
    requested = (os.environ.get(API_EFFORT_ENV) or "").strip().lower()
    if requested and requested != RESEARCH_EFFORT:
        raise ValueError(
            f"{API_EFFORT_ENV} must be {RESEARCH_EFFORT} for this skill; got {requested}"
        )
    return RESEARCH_EFFORT


def manifest_fields(declared=None):
    """Audit fields describing the research role of the current run."""
    return {
        "research_provider": SESSION_PROVIDER,
        "research_model": session_model(declared),
        "model_provenance": SESSION_MODEL_PROVENANCE,
        "reasoning_effort_requested": None,
        "reasoning_effort_effective": SESSION_EFFORT_STATUS,
        "standalone_model_api": False,
    }
