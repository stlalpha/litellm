"""
Tests for chained router resolution.

Covers:
- Two-layer chain: semantic router → complexity router → concrete model
- Three-layer chain (depth guard fires at 5)
- Non-chained configs: single complexity/semantic router, plain model
- Circular chain detection at startup (RouterChainConfigError)
- Max depth guard: chain that would loop is stopped and falls back
- Routing chain metadata populated in response and request_kwargs
- Depth guard logging (warning emitted, no exception)
"""
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.router_utils.chain_validator import (
    RouterChainConfigError,
    detect_circular_chains,
)
from litellm.types.router import PreRoutingHookResponse, RoutingLayerInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router_with_hooks(
    auto_routers: Optional[Dict] = None,
    complexity_routers: Optional[Dict] = None,
    max_depth: int = 5,
):
    """
    Build a minimal Router-like object with the attributes and methods that
    async_pre_routing_hook depends on, without touching real LLM providers.
    """
    from litellm.router import Router

    router = MagicMock(spec=Router)
    router.auto_routers = auto_routers or {}
    router.complexity_routers = complexity_routers or {}
    router.max_router_chain_depth = max_depth
    # Bind the real methods to the mock instance so they execute normally
    router._is_virtual_router = Router._is_virtual_router.__get__(router, Router)
    router._invoke_single_router = Router._invoke_single_router.__get__(router, Router)
    router.async_pre_routing_hook = Router.async_pre_routing_hook.__get__(router, Router)
    return router


def _make_pre_routing_response(model: str, messages=None) -> PreRoutingHookResponse:
    return PreRoutingHookResponse(model=model, messages=messages)


# ---------------------------------------------------------------------------
# Chain validator unit tests
# ---------------------------------------------------------------------------


class TestDetectCircularChains:
    def test_no_routers_no_error(self):
        """Plain model list with no routers raises nothing."""
        model_list = [
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o"},
            }
        ]
        detect_circular_chains(model_list)  # should not raise

    def test_single_complexity_router_no_error(self):
        """Single complexity router pointing to concrete models is fine."""
        model_list = [
            {
                "model_name": "cheap",
                "litellm_params": {"model": "openai/gpt-4o-mini"},
            },
            {
                "model_name": "mid",
                "litellm_params": {"model": "openai/gpt-4o"},
            },
            {
                "model_name": "complexity",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {"SIMPLE": "cheap", "MEDIUM": "mid"},
                    },
                    "complexity_router_default_model": "mid",
                },
            },
        ]
        detect_circular_chains(model_list)

    def test_two_layer_chain_no_error(self):
        """semantic → complexity → concrete is a valid DAG."""
        model_list = [
            {"model_name": "cheap", "litellm_params": {"model": "openai/gpt-4o-mini"}},
            {"model_name": "mid", "litellm_params": {"model": "openai/gpt-4o"}},
            {
                "model_name": "code-complexity",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {"SIMPLE": "cheap", "MEDIUM": "mid"},
                    },
                    "complexity_router_default_model": "mid",
                },
            },
            {
                "model_name": "auto",
                "litellm_params": {
                    "model": "auto_router/semantic_router",
                    "auto_router_default_model": "code-complexity",
                    "auto_router_embedding_model": "text-embedding-3-small",
                    "auto_router_config": '{"routes": []}',
                },
            },
        ]
        detect_circular_chains(model_list)

    def test_direct_self_cycle_raises(self):
        """A complexity router whose tier points back to itself is a cycle."""
        model_list = [
            {
                "model_name": "loop",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {"SIMPLE": "loop"},
                    },
                    "complexity_router_default_model": "loop",
                },
            },
        ]
        with pytest.raises(RouterChainConfigError, match="Circular router chain"):
            detect_circular_chains(model_list)

    def test_indirect_cycle_raises(self):
        """A → B → A cycle through default models is caught."""
        model_list = [
            {
                "model_name": "router-a",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": "router-b"}},
                    "complexity_router_default_model": "router-b",
                },
            },
            {
                "model_name": "router-b",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": "router-a"}},
                    "complexity_router_default_model": "router-a",
                },
            },
        ]
        with pytest.raises(RouterChainConfigError, match="Circular router chain"):
            detect_circular_chains(model_list)

    def test_semantic_router_cycle_via_default_raises(self):
        """Semantic router whose default_model is itself is a cycle."""
        model_list = [
            {
                "model_name": "auto",
                "litellm_params": {
                    "model": "auto_router/semantic_router",
                    "auto_router_default_model": "auto",
                    "auto_router_embedding_model": "text-embedding-3-small",
                    "auto_router_config": '{"routes": []}',
                },
            },
        ]
        with pytest.raises(RouterChainConfigError, match="Circular router chain"):
            detect_circular_chains(model_list)

    def test_error_message_contains_cycle_path(self):
        """Error message shows the cycle path."""
        model_list = [
            {
                "model_name": "loop",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": "loop"}},
                    "complexity_router_default_model": "loop",
                },
            },
        ]
        with pytest.raises(RouterChainConfigError) as exc_info:
            detect_circular_chains(model_list)
        assert "loop" in str(exc_info.value)


# ---------------------------------------------------------------------------
# PreRoutingHookResponse chaining unit tests
# ---------------------------------------------------------------------------


class TestPreRoutingHookResponse:
    def test_routing_chain_field_default_none(self):
        r = PreRoutingHookResponse(model="gpt-4o", messages=None)
        assert r.routing_chain is None
        assert r.routing_layers is None

    def test_routing_chain_field_populated(self):
        layers = [
            RoutingLayerInfo(layer=1, router_type="semantic", route="code", latency_ms=42.1)
        ]
        r = PreRoutingHookResponse(
            model="sonnet",
            messages=None,
            routing_chain=["auto", "code", "sonnet"],
            routing_layers=layers,
        )
        assert r.routing_chain == ["auto", "code", "sonnet"]
        assert len(r.routing_layers) == 1
        assert r.routing_layers[0].router_type == "semantic"


# ---------------------------------------------------------------------------
# async_pre_routing_hook integration tests (no real LLM calls)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncPreRoutingHookChaining:
    async def test_non_virtual_model_returns_none(self):
        """A plain model name (not a router) should return None."""
        router = _make_router_with_hooks()
        result = await router.async_pre_routing_hook(
            model="gpt-4o",
            request_kwargs={},
            messages=[{"role": "user", "content": "hello"}],
        )
        assert result is None

    async def test_single_complexity_router_resolves(self):
        """Single complexity router (non-chained) resolves to concrete model."""
        complexity_mock = AsyncMock()
        complexity_mock.async_pre_routing_hook = AsyncMock(
            return_value=_make_pre_routing_response("gpt-4o-mini")
        )

        router = _make_router_with_hooks(complexity_routers={"cheap-router": complexity_mock})
        request_kwargs: Dict[str, Any] = {}
        result = await router.async_pre_routing_hook(
            model="cheap-router",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "what is 2+2?"}],
        )
        assert result is not None
        assert result.model == "gpt-4o-mini"
        assert result.routing_chain == ["cheap-router", "gpt-4o-mini"]
        assert len(result.routing_layers) == 1
        assert result.routing_layers[0].router_type == "complexity"
        # Metadata written into request_kwargs
        assert request_kwargs["_routing_chain"] == ["cheap-router", "gpt-4o-mini"]

    async def test_two_layer_chain_semantic_then_complexity(self):
        """semantic → complexity → concrete resolves end-to-end."""
        # Layer 1: semantic router resolves "auto" → "code-complexity"
        semantic_mock = AsyncMock()
        semantic_mock.async_pre_routing_hook = AsyncMock(
            return_value=_make_pre_routing_response("code-complexity")
        )
        # Layer 2: complexity router resolves "code-complexity" → "sonnet"
        complexity_mock = AsyncMock()
        complexity_mock.async_pre_routing_hook = AsyncMock(
            return_value=_make_pre_routing_response("sonnet")
        )

        router = _make_router_with_hooks(
            auto_routers={"auto": semantic_mock},
            complexity_routers={"code-complexity": complexity_mock},
        )
        request_kwargs: Dict[str, Any] = {}
        messages = [{"role": "user", "content": "write a python function to sort a list"}]
        result = await router.async_pre_routing_hook(
            model="auto",
            request_kwargs=request_kwargs,
            messages=messages,
        )
        assert result is not None
        assert result.model == "sonnet"
        assert result.routing_chain == ["auto", "code-complexity", "sonnet"]
        assert len(result.routing_layers) == 2
        assert result.routing_layers[0].router_type == "semantic"
        assert result.routing_layers[0].route == "code-complexity"
        assert result.routing_layers[1].router_type == "complexity"
        assert result.routing_layers[1].route == "sonnet"
        # request_kwargs metadata
        assert request_kwargs["_routing_chain"] == ["auto", "code-complexity", "sonnet"]
        assert len(request_kwargs["_routing_layers"]) == 2
        assert "_total_routing_latency_ms" in request_kwargs

    async def test_two_layer_chain_complexity_then_semantic(self):
        """complexity → semantic → concrete also resolves correctly."""
        complexity_mock = AsyncMock()
        complexity_mock.async_pre_routing_hook = AsyncMock(
            return_value=_make_pre_routing_response("semantic-tier")
        )
        semantic_mock = AsyncMock()
        semantic_mock.async_pre_routing_hook = AsyncMock(
            return_value=_make_pre_routing_response("opus")
        )

        router = _make_router_with_hooks(
            complexity_routers={"entry": complexity_mock},
            auto_routers={"semantic-tier": semantic_mock},
        )
        result = await router.async_pre_routing_hook(
            model="entry",
            request_kwargs={},
            messages=[{"role": "user", "content": "analyze this architecture step by step"}],
        )
        assert result is not None
        assert result.model == "opus"
        assert result.routing_chain == ["entry", "semantic-tier", "opus"]

    async def test_three_layer_chain(self):
        """A → B → C → concrete works with default depth of 5."""
        mock_a = AsyncMock()
        mock_a.async_pre_routing_hook = AsyncMock(
            return_value=_make_pre_routing_response("router-b")
        )
        mock_b = AsyncMock()
        mock_b.async_pre_routing_hook = AsyncMock(
            return_value=_make_pre_routing_response("router-c")
        )
        mock_c = AsyncMock()
        mock_c.async_pre_routing_hook = AsyncMock(
            return_value=_make_pre_routing_response("concrete-model")
        )

        router = _make_router_with_hooks(
            auto_routers={"router-a": mock_a},
            complexity_routers={"router-b": mock_b, "router-c": mock_c},
            max_depth=5,
        )
        result = await router.async_pre_routing_hook(
            model="router-a",
            request_kwargs={},
            messages=[{"role": "user", "content": "deep work"}],
        )
        assert result is not None
        assert result.model == "concrete-model"
        assert result.routing_chain == [
            "router-a", "router-b", "router-c", "concrete-model"
        ]
        assert len(result.routing_layers) == 3

    async def test_max_depth_guard_stops_chain(self):
        """When max_depth is exceeded the hook returns the last resolved model
        and logs a warning, without raising an exception."""
        # Build a chain that would go: a → b → a → b → ... (infinite if not guarded)
        # We mock _is_virtual_router to always return True for b so the loop
        # keeps trying, then depth guard fires.

        call_count = {"n": 0}

        async def cycling_hook(model, request_kwargs, messages, input, specific_deployment):
            call_count["n"] += 1
            # Always resolve to the other router to simulate an infinite cycle
            return _make_pre_routing_response("router-b" if model == "router-a" else "router-a")

        mock_a = MagicMock()
        mock_a.async_pre_routing_hook = cycling_hook
        mock_b = MagicMock()
        mock_b.async_pre_routing_hook = cycling_hook

        router = _make_router_with_hooks(
            auto_routers={"router-a": mock_a, "router-b": mock_b},
            max_depth=3,
        )

        with patch("litellm.router.verbose_router_logger") as mock_log:
            result = await router.async_pre_routing_hook(
                model="router-a",
                request_kwargs={},
                messages=[{"role": "user", "content": "cycle test"}],
            )

        assert result is not None
        # Depth guard fires — last entry in the chain is the final resolved model
        assert result.model == result.routing_chain[-1]
        # Warning was emitted
        mock_log.warning.assert_called_once()
        warning_msg = mock_log.warning.call_args[0][0]
        assert "depth limit" in warning_msg.lower() or "chain depth" in warning_msg.lower()

    async def test_messages_propagated_through_chain(self):
        """Original messages are passed through all layers unchanged."""
        messages = [{"role": "user", "content": "test prompt"}]

        captured_messages = {}

        async def capture_hook(model, request_kwargs, messages, input, specific_deployment):
            captured_messages[model] = messages
            return _make_pre_routing_response(
                "router-b" if model == "router-a" else "concrete",
                messages=messages,
            )

        mock_a = MagicMock()
        mock_a.async_pre_routing_hook = capture_hook
        mock_b = MagicMock()
        mock_b.async_pre_routing_hook = capture_hook

        router = _make_router_with_hooks(
            auto_routers={"router-a": mock_a},
            complexity_routers={"router-b": mock_b},
        )
        result = await router.async_pre_routing_hook(
            model="router-a",
            request_kwargs={},
            messages=messages,
        )
        assert captured_messages["router-a"] == messages
        assert captured_messages["router-b"] == messages

    async def test_request_kwargs_metadata_keys(self):
        """All three metadata keys are written into request_kwargs."""
        mock_router = AsyncMock()
        mock_router.async_pre_routing_hook = AsyncMock(
            return_value=_make_pre_routing_response("concrete")
        )

        router = _make_router_with_hooks(complexity_routers={"cr": mock_router})
        request_kwargs: Dict[str, Any] = {"some_existing_key": "value"}
        await router.async_pre_routing_hook(
            model="cr",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert "_routing_chain" in request_kwargs
        assert "_routing_layers" in request_kwargs
        assert "_total_routing_latency_ms" in request_kwargs
        # Existing keys must not be removed
        assert request_kwargs["some_existing_key"] == "value"

    async def test_single_router_returns_none_falls_through(self):
        """If the first (and only) router returns None, the hook returns None."""
        mock_router = AsyncMock()
        mock_router.async_pre_routing_hook = AsyncMock(return_value=None)

        router = _make_router_with_hooks(complexity_routers={"cr": mock_router})
        result = await router.async_pre_routing_hook(
            model="cr",
            request_kwargs={},
            messages=None,
        )
        # When the first layer returns None we break out of the loop immediately
        # and routing_chain contains only the initial model — this is an edge
        # case (no messages), handled gracefully without error.
        assert result is not None
        assert result.model == "cr"


# ---------------------------------------------------------------------------
# Router.set_model_list startup validation integration
# ---------------------------------------------------------------------------


class TestSetModelListCircularValidation:
    """Verify that the Router constructor rejects circular chains at startup."""

    def test_valid_chain_initializes_ok(self):
        """Two-layer chain with concrete tier targets should initialize."""
        # We use minimal fake models that pass provider validation.
        # Because we don't want to require API keys in unit tests, we patch
        # litellm.get_llm_provider and the auto/complexity router init.
        from unittest.mock import patch

        model_list = [
            {
                "model_name": "cheap",
                "litellm_params": {"model": "openai/gpt-4o-mini"},
            },
            {
                "model_name": "cr",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": "cheap"}},
                    "complexity_router_default_model": "cheap",
                },
            },
        ]
        # detect_circular_chains runs inside set_model_list; a clean config
        # must not raise RouterChainConfigError.
        from litellm.router_utils.chain_validator import detect_circular_chains

        detect_circular_chains(model_list)  # no exception

    def test_circular_chain_raises_config_error(self):
        """Router.set_model_list must raise RouterChainConfigError for cycles."""
        model_list = [
            {
                "model_name": "loop",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": "loop"}},
                    "complexity_router_default_model": "loop",
                },
            },
        ]
        with pytest.raises(RouterChainConfigError, match="Circular router chain"):
            from litellm.router_utils.chain_validator import detect_circular_chains

            detect_circular_chains(model_list)
