"""Model registry invariants."""

from agent_bluff.config import DEFAULT_FIELD, MODELS, POC_FIELD, RunConfig


def test_every_field_resolves():
    for field in (DEFAULT_FIELD, POC_FIELD):
        assert RunConfig(models=field).specs()


def test_poc_field_covers_the_same_providers_as_the_real_field():
    """A dry run is only useful if it exercises the same provider code paths."""
    providers = {MODELS[k].id.split("/")[0] for k in DEFAULT_FIELD}
    assert {MODELS[k].id.split("/")[0] for k in POC_FIELD} == providers


def test_poc_includes_the_anthropic_cache_control_path():
    """That branch is provider-specific and would otherwise go untested."""
    assert any(MODELS[k].explicit_cache_control for k in POC_FIELD)


def test_unknown_model_keys_are_rejected_with_a_useful_message():
    import pytest

    with pytest.raises(KeyError, match="nonexistent"):
        RunConfig(models=("nonexistent",)).specs()


def test_open_weight_models_are_provider_pinned():
    """Multiple hosts at differing quantisations would make a run irreproducible."""
    for key in ("deepseek", "qwen"):
        assert MODELS[key].provider_order, f"{key} must pin its provider"
    for key in DEFAULT_FIELD + POC_FIELD:
        assert not MODELS[key].provider_order, f"{key} is closed-weight; pinning is redundant"


def test_ordered_pairs_exclude_self_play():
    pairs = RunConfig(models=("a", "b", "c")).ordered_pairs()
    assert len(pairs) == 6
    assert all(i != c for i, c in pairs)

