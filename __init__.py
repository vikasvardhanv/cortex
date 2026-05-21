"""
CORTEX: Computational ORchestration for Technical Engineering eXecution

A computational engineering model (CEM) that orchestrates physics solvers,
ML models, and optimization engines to generate optimized engineering designs.

Inspired by Leap71's Noyron approach to computational engineering.
"""

__version__ = "0.1.0"

from knowledge import MaterialsDB, PhysicsRules
from pipeline import CortexEngine

__all__ = [
    "MaterialsDB",
    "PhysicsRules",
    "CortexEngine",
    "__version__",
]
