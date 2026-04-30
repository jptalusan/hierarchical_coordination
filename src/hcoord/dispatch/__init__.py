"""Dispatchers and the shared insertion primitive."""

from hcoord.dispatch.base import DispatchResult, Dispatcher
from hcoord.dispatch.hierarchical import HierarchicalDispatcher
from hcoord.dispatch.insertion import InsertionResult, apply_insertion, best_insertion
from hcoord.dispatch.monolithic import MonolithicDispatcher

__all__ = [
    "DispatchResult",
    "Dispatcher",
    "HierarchicalDispatcher",
    "InsertionResult",
    "MonolithicDispatcher",
    "apply_insertion",
    "best_insertion",
]
