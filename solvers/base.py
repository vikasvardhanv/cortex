"""
Base classes for Cortex solvers.

All solvers inherit from these base classes to ensure
consistent interface across different physics types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np


@dataclass
class Problem:
    """
    Base class for problem definitions.

    A problem contains:
    - Geometry/domain definition
    - Material properties
    - Boundary conditions
    - Initial conditions (for transient)
    """
    name: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Dict[str, Any]:
        """Validate problem setup. Returns dict with 'valid' and 'errors'."""
        return {"valid": True, "errors": [], "warnings": []}


@dataclass
class Result:
    """
    Base class for solver results.

    Contains:
    - Solution fields (temperature, displacement, etc.)
    - Metadata (solve time, iterations, etc.)
    - Validation status
    """
    problem_name: str
    solver_name: str
    solve_time: float  # seconds
    converged: bool
    iterations: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def summary(self) -> str:
        """Return a human-readable summary of results."""
        pass

    @abstractmethod
    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        """Get a solution field by name."""
        pass


class Solver(ABC):
    """
    Abstract base class for all Cortex solvers.

    Solvers must implement:
    - solve(problem) -> Result
    - validate_problem(problem) -> bool
    """

    def __init__(self, name: str = "BaseSolver"):
        self.name = name
        self._solve_count = 0

    @abstractmethod
    def solve(self, problem: Problem) -> Result:
        """
        Solve the given problem.

        Args:
            problem: Problem definition

        Returns:
            Result object containing solution
        """
        pass

    @abstractmethod
    def validate_problem(self, problem: Problem) -> Dict[str, Any]:
        """
        Validate that the problem is properly defined.

        Returns:
            Dict with 'valid' (bool), 'errors' (list), 'warnings' (list)
        """
        pass

    def can_solve(self, problem: Problem) -> bool:
        """Check if this solver can handle the given problem type."""
        validation = self.validate_problem(problem)
        return validation.get("valid", False)

    def info(self) -> Dict[str, Any]:
        """Return information about this solver."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "solves_completed": self._solve_count,
        }
