"""QuakeCore tool registry — auto-discovery and self-registration.

Usage:
    from quakecore_tools.registry import register_tool

    @register_tool(
        name="my_tool",
        category="analysis",
        description="Run custom analysis",
        triggers=["自定义分析", "custom analysis"],
    )
    def my_tool(params=None):
        ...

To add a new tool, just create a file in quakecore_tools/ with @register_tool.
No other files need to be modified.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolMeta:
    """Metadata for a registered tool."""

    name: str
    func: Callable[..., Any]
    description: str
    category: str
    triggers: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    needs_file: bool = False
    file_types: list[str] = field(default_factory=list)


_REGISTRY: dict[str, ToolMeta] = {}


def register_tool(
    *,
    name: str = "",
    category: str = "general",
    description: str = "",
    triggers: list[str] | None = None,
    examples: list[str] | None = None,
    needs_file: bool = False,
    file_types: list[str] | None = None,
):
    """Decorator: register a tool into the global registry.

    The decorated function must be a LangChain @tool-compatible callable
    (i.e., accept ``params`` and return a JSON string).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        doc = description or (func.__doc__ or "").strip().split("\n")[0]
        meta = ToolMeta(
            name=tool_name,
            func=func,
            description=doc,
            category=category,
            triggers=triggers or [],
            examples=examples or [],
            needs_file=needs_file,
            file_types=file_types or [],
        )
        _REGISTRY[meta.name] = meta
        return func

    return decorator


def get_registry() -> dict[str, ToolMeta]:
    """Return a copy of the current registry."""
    return dict(_REGISTRY)


def get_tool(name: str) -> ToolMeta | None:
    """Look up a single tool by name."""
    return _REGISTRY.get(name)


def get_tools_by_category(category: str) -> list[ToolMeta]:
    """Return all tools in a given category."""
    return [m for m in _REGISTRY.values() if m.category == category]


def build_tool_list() -> list[Callable[..., Any]]:
    """Return a list of tool callables for the LangChain agent."""
    return [meta.func for meta in _REGISTRY.values()]


def build_tool_descriptions() -> str:
    """Generate a human-readable tool catalogue for AI prompts.

    Tools are grouped by category with their description, triggers, and examples.
    """
    categories: dict[str, list[ToolMeta]] = {}
    for meta in _REGISTRY.values():
        categories.setdefault(meta.category, []).append(meta)

    lines: list[str] = []
    for cat in sorted(categories):
        tools = categories[cat]
        lines.append(f"\n### {cat.upper()} TOOLS")
        for t in tools:
            lines.append(f"- **{t.name}**: {t.description}")
            if t.triggers:
                lines.append(f"  Triggers: {', '.join(t.triggers[:6])}")
            if t.examples:
                lines.append(f"  Example: {t.examples[0]}")
            if t.needs_file:
                ft = ", ".join(t.file_types) if t.file_types else "any"
                lines.append(f"  Requires: loaded file ({ft})")
    return "\n".join(lines)


def build_tool_names() -> list[str]:
    """Return a list of registered tool names."""
    return list(_REGISTRY.keys())


def auto_discover(package_name: str = "quakecore_tools") -> None:
    """Import all modules in *package_name* to trigger @register_tool decorators.

    Call this once at startup (e.g., in ``get_agent_executor``).
    """
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        if module_name.startswith("_") or module_name == "registry":
            continue
        try:
            importlib.import_module(f"{package_name}.{module_name}")
        except Exception:
            # Don't let a broken module prevent other tools from loading.
            pass
