"""Base Plugin Contract and Abstract Interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from engine.schemas import PluginContextResponse, PluginManifest


class BasePlugin(ABC):
    """Abstract interface defining the contract for trusted in-process plugins.

    Security and Trust Boundaries:
    - Trusted plugin implementations run within host Python process context.
    - Plugins receive an immutable, typed PluginContextResponse payload.
    - Plugins do NOT receive raw SQLite connections, GPU semaphores, or internal models.
    - Output must be a dictionary serializable to standard JSON.
    """

    @property
    @abstractmethod
    def manifest(self) -> PluginManifest:
        """Return the immutable manifest describing this plugin."""
        pass

    @abstractmethod
    def run(self, context: PluginContextResponse, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute plugin analysis over the provided recording context.

        Parameters
        ----------
        context : PluginContextResponse
            Consolidated recording and transcript context.
        parameters : dict[str, Any]
            Validated hyperparameter options for the run.

        Returns
        -------
        dict[str, Any]
            Structured result payload (e.g. lexical analysis, speaker metrics, annotations).
        """
        pass
