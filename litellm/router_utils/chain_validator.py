"""
Startup validation for chained router configurations.

Detects circular references in router chains before they cause infinite loops
at request time.
"""
from typing import Any, Dict, List, Set


class RouterChainConfigError(ValueError):
    """Raised when a circular router chain is detected during startup validation."""


def _build_full_graph(model_list: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Build a directed graph of virtual model -> [possible target model names].

    For complexity routers every configured tier target is an edge.
    For semantic routers the default model is the edge (routes resolved at
    request-time are not available statically, but the default covers the
    fallback path which is the common cycle risk).
    """
    graph: Dict[str, List[str]] = {}

    for entry in model_list:
        model_name: str = entry.get("model_name") or ""
        litellm_params: Dict[str, Any] = entry.get("litellm_params") or {}
        model: str = litellm_params.get("model") or ""

        if not model_name:
            continue

        if model.startswith("auto_router/complexity_router"):
            config: Dict[str, Any] = (
                litellm_params.get("complexity_router_config") or {}
            )
            tiers: Dict[str, str] = config.get("tiers") or {}
            default: str = litellm_params.get("complexity_router_default_model") or ""
            edges: List[str] = [t for t in tiers.values() if t]
            if default and default not in edges:
                edges.append(default)
            graph[model_name] = edges

        elif model.startswith("auto_router/"):
            default = litellm_params.get("auto_router_default_model") or ""
            graph[model_name] = [default] if default else []

    return graph


def detect_circular_chains(model_list: List[Dict[str, Any]]) -> None:
    """
    Validate that no circular router chains exist in model_list.

    Performs DFS cycle detection on the directed graph where nodes are virtual
    auto-router model names and edges are their configured targets.

    Raises:
        RouterChainConfigError: If a cycle is found, with a clear description
            of the cycle path.
    """
    graph = _build_full_graph(model_list)
    virtual_nodes: Set[str] = set(graph.keys())

    visited: Set[str] = set()
    in_stack: Set[str] = set()

    def dfs(node: str, path: List[str]) -> None:
        if node not in virtual_nodes:
            return  # concrete deployment, no outgoing edges
        if node in in_stack:
            cycle_start = path.index(node)
            cycle_path = path[cycle_start:] + [node]
            raise RouterChainConfigError(
                "Circular router chain detected: "
                + " → ".join(cycle_path)
                + "\nFix: ensure no auto_router target references a model "
                "that eventually routes back to itself."
            )
        if node in visited:
            return
        visited.add(node)
        in_stack.add(node)
        for neighbor in graph.get(node, []):
            dfs(neighbor, path + [node])
        in_stack.discard(node)

    for node in virtual_nodes:
        if node not in visited:
            dfs(node, [])
