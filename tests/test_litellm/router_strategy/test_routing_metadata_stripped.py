"""
Test that routing chain metadata is stripped before outbound API calls.

The chained router resolution writes _routing_chain, _routing_layers,
_total_routing_latency_ms, and max_router_chain_depth into kwargs for
internal observability. These must not leak to backend APIs (OpenAI,
Anthropic, etc.) which reject unknown parameters with 400 errors.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.router import Router


ROUTING_METADATA_KEYS = (
    "_routing_chain",
    "_routing_layers",
    "_total_routing_latency_ms",
    "max_router_chain_depth",
)


@pytest.mark.asyncio
async def test_routing_metadata_not_in_acompletion_call():
    """Verify that routing chain metadata in kwargs is stripped before
    the final litellm.acompletion() call, so backend APIs never see it."""

    model_list = [
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "sk-fake-key-for-testing",
            },
        },
    ]
    router = Router(model_list=model_list)

    captured_kwargs = {}

    async def mock_acompletion(*args, **kwargs):
        captured_kwargs.update(kwargs)
        # Return a minimal mock response
        mock_response = MagicMock()
        mock_response.model = "gpt-4o-mini"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test"
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10)
        mock_response._hidden_params = {}
        return mock_response

    with patch.object(litellm, "acompletion", side_effect=mock_acompletion):
        # Simulate what happens after chained routing populates metadata
        await router._acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            _routing_chain=["auto", "code", "test-model"],
            _routing_layers=[{"layer": 1, "router_type": "semantic", "route": "code", "latency_ms": 1.5}],
            _total_routing_latency_ms=2.3,
            max_router_chain_depth=5,
        )

    for key in ROUTING_METADATA_KEYS:
        assert key not in captured_kwargs, (
            f"Routing metadata key '{key}' leaked into outbound acompletion call. "
            f"Backend APIs will reject this as an unknown parameter."
        )


@pytest.mark.asyncio
async def test_routing_metadata_stripped_from_litellm_params():
    """Verify that max_router_chain_depth in litellm_params (from config)
    is also stripped before the outbound call."""

    model_list = [
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "sk-fake-key-for-testing",
                "max_router_chain_depth": 5,  # This comes from config
            },
        },
    ]
    router = Router(model_list=model_list)

    captured_kwargs = {}

    async def mock_acompletion(*args, **kwargs):
        captured_kwargs.update(kwargs)
        mock_response = MagicMock()
        mock_response.model = "gpt-4o-mini"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test"
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10)
        mock_response._hidden_params = {}
        return mock_response

    with patch.object(litellm, "acompletion", side_effect=mock_acompletion):
        await router._acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert "max_router_chain_depth" not in captured_kwargs, (
        "max_router_chain_depth from litellm_params leaked into outbound call"
    )
