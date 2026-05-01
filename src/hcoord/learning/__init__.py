"""Learning utilities for the hierarchical dispatcher.

v1: cost-approximation training data collection for the LLP. The collector
hooks into `best_insertion` via an observer callback and records features
per (vehicle, request) pair plus the ground-truth best feasible cost.
"""

from hcoord.learning.collector import InsertionCollector, InsertionRow
from hcoord.learning.features import extract_features

__all__ = ["InsertionCollector", "InsertionRow", "extract_features"]
