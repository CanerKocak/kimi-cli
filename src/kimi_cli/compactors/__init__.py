from __future__ import annotations

import importlib
from collections.abc import Callable

from kimi_cli.llm import LLM
from kimi_cli.soul.compaction import BaseCompactor, SimpleCompaction

from .morph import MorphCompaction


def _looks_like_morph_llm(llm: LLM | None) -> bool:
    if llm is None:
        return False
    provider = llm.provider_config
    if provider is None or provider.type not in {"openai_legacy", "openai_responses"}:
        return False

    base_url = provider.base_url.lower()
    model_name = (
        llm.model_config.model.lower() if llm.model_config is not None else llm.model_name.lower()
    )
    return "morph" in base_url or "morph" in model_name


_BUILTIN_COMPACTORS: dict[str, Callable[[], BaseCompactor]] = {
    "simple": SimpleCompaction,
    "morph": MorphCompaction,
}


def load_compactor(provider_name: str | None, llm: LLM | None) -> BaseCompactor:
    """Create the compactor selected by config or inferred from the compaction LLM."""
    if provider_name is None:
        return MorphCompaction() if _looks_like_morph_llm(llm) else SimpleCompaction()

    if provider_name in _BUILTIN_COMPACTORS:
        return _BUILTIN_COMPACTORS[provider_name]()

    module_name, _, class_name = provider_name.rpartition(".")
    if not module_name or not class_name:
        builtin_names = ", ".join(sorted(_BUILTIN_COMPACTORS))
        raise ValueError(
            f"Unknown compaction provider '{provider_name}'. Use one of: {builtin_names}, "
            "or a full import path."
        )

    module = importlib.import_module(module_name)
    compactor_cls = getattr(module, class_name)
    compactor = compactor_cls()
    if not isinstance(compactor, BaseCompactor):
        raise TypeError(
            f"Compaction provider '{provider_name}' must implement BaseCompactor.compact(...)."
        )
    return compactor


__all__ = ["MorphCompaction", "load_compactor"]
