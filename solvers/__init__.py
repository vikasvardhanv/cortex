"""
Physics Solvers for Cortex CEM

Contains numerical solvers for:
- Thermal analysis (heat conduction, convection)
- Structural analysis (FEM stress/strain)
- Fluid dynamics (Lattice Boltzmann CFD)
- Coupled multi-physics
- Design optimization (genetic algorithms)
"""

from .base import Solver, Problem, Result
from .thermal import ThermalSolver, ThermalProblem, ThermalResult, BoundaryCondition
from .structural import (
    StructuralSolver,
    StructuralProblem,
    StructuralResult,
    StructuralBoundaryCondition,
)
from .fluid import (
    LBMSolver,
    FluidProblem,
    FluidResult,
    FluidBoundaryCondition,
    ChannelFlowBenchmark,
    CylinderFlowBenchmark,
)
from .optimizer import (
    GeneticAlgorithm,
    NSGA2,
    DifferentialEvolution,
    DesignOptimizer,
    OptimizationProblem,
    OptimizationResult,
    DesignVariable,
    Constraint,
)
from .coupled import (
    CoupledSolver,
    CoupledProblem,
    CoupledResult,
    FluidStructureInteraction,
)

__all__ = [
    # Base
    "Solver",
    "Problem",
    "Result",
    # Thermal
    "ThermalSolver",
    "ThermalProblem",
    "ThermalResult",
    "BoundaryCondition",
    # Structural
    "StructuralSolver",
    "StructuralProblem",
    "StructuralResult",
    "StructuralBoundaryCondition",
    # Fluid
    "LBMSolver",
    "FluidProblem",
    "FluidResult",
    "FluidBoundaryCondition",
    "ChannelFlowBenchmark",
    "CylinderFlowBenchmark",
    # Optimization
    "GeneticAlgorithm",
    "NSGA2",
    "DifferentialEvolution",
    "DesignOptimizer",
    "OptimizationProblem",
    "OptimizationResult",
    "DesignVariable",
    "Constraint",
    # Coupled
    "CoupledSolver",
    "CoupledProblem",
    "CoupledResult",
    "FluidStructureInteraction",
]
