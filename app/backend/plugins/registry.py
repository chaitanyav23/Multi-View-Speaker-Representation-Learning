"""Explicit Host-Controlled Plugin Registry."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional
from engine.schemas import PluginManifest
from .base import BasePlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Central in-process registry for trusted plugin implementations.

    Safety Rules:
    - Zero dynamic execution of arbitrary files or path-based imports.
    - All plugins must explicitly register an instance implementing BasePlugin.
    - Duplicate plugin IDs are strictly rejected to prevent hijacking.
    """

    _plugins: Dict[str, BasePlugin] = {}

    @classmethod
    def register(cls, plugin: BasePlugin) -> None:
        """Register a trusted plugin implementation.

        Parameters
        ----------
        plugin : BasePlugin
            Instance of a class implementing BasePlugin.
        """
        if not isinstance(plugin, BasePlugin):
            raise TypeError(f"Plugin must inherit from BasePlugin, got {type(plugin).__name__}")

        manifest = plugin.manifest
        if not isinstance(manifest, PluginManifest):
            raise TypeError(f"Plugin manifest must be an instance of PluginManifest, got {type(manifest).__name__}")

        plugin_id = manifest.plugin_id
        if plugin_id in cls._plugins:
            raise ValueError(f"Plugin with ID '{plugin_id}' is already registered.")

        cls._plugins[plugin_id] = plugin
        logger.info("Registered plugin: %s v%s (%s)", manifest.name, manifest.version, plugin_id)

    @classmethod
    def unregister(cls, plugin_id: str) -> Optional[BasePlugin]:
        """Unregister a plugin by ID."""
        return cls._plugins.pop(plugin_id, None)

    @classmethod
    def get(cls, plugin_id: str) -> Optional[BasePlugin]:
        """Retrieve a registered plugin by ID."""
        return cls._plugins.get(plugin_id)

    @classmethod
    def list_manifests(cls) -> List[PluginManifest]:
        """Return list of manifests for all registered plugins."""
        return [p.manifest for p in cls._plugins.values() if p.manifest.enabled]

    @classmethod
    def list_all_manifests(cls) -> List[PluginManifest]:
        """Return list of manifests for all registered plugins regardless of enabled status."""
        return [p.manifest for p in cls._plugins.values()]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered plugins (primarily used for test teardown)."""
        cls._plugins.clear()
