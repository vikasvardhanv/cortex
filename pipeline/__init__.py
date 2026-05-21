"""
Pipeline Module for Cortex CEM

Orchestrates the flow from natural language input to geometry output:
1. Router: Parses input and determines required solvers
2. Executor: Runs solvers in correct order
3. Validator: Checks results against engineering constraints
"""

from .router import Router, ProblemSpec
from .executor import PipelineExecutor, Pipeline
from .engine import CortexEngine

__all__ = [
    "Router",
    "ProblemSpec",
    "PipelineExecutor",
    "Pipeline",
    "CortexEngine",
]
