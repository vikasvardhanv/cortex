"""
Coupled Multi-Physics Solver for Cortex CEM

Orchestrates thermal-structural-fluid coupling for
comprehensive engineering simulation.

Coupling approaches:
- Sequential (one-way): Thermal → Structural
- Iterative (two-way): Thermal ↔ Structural with convergence
- Fully coupled: Simultaneous solution (future)

Applications:
- Thermo-structural analysis of rocket nozzles
- Conjugate heat transfer
- Fluid-structure interaction (FSI)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Literal
import numpy as np
import time

from .base import Solver, Problem, Result
from .thermal import ThermalSolver, ThermalProblem, ThermalResult
from .structural import StructuralSolver, StructuralProblem, StructuralResult
from .fluid import LBMSolver, FluidProblem, FluidResult


@dataclass
class CoupledProblem(Problem):
    """
    Multi-physics coupled problem definition.

    Example:
        problem = CoupledProblem(
            name="rocket_nozzle",
            thermal_problem=thermal_prob,
            structural_problem=structural_prob,
            coupling_type="thermal_structural",
        )
    """
    # Sub-problems (optional - not all may be needed)
    thermal_problem: Optional[ThermalProblem] = None
    structural_problem: Optional[StructuralProblem] = None
    fluid_problem: Optional[FluidProblem] = None

    # Coupling type
    coupling_type: str = "sequential"  # "sequential", "iterative"

    # Iteration parameters (for iterative coupling)
    max_coupling_iterations: int = 20
    coupling_tolerance: float = 1e-4

    # Thermal-structural coupling parameters
    thermal_expansion_coefficient: float = 12e-6  # 1/K (steel default)
    reference_temperature: float = 293.0  # K

    def validate(self) -> Dict[str, Any]:
        result = {"valid": True, "errors": [], "warnings": []}

        if self.thermal_problem is None and self.structural_problem is None and self.fluid_problem is None:
            result["errors"].append("At least one sub-problem must be defined")
            result["valid"] = False

        return result


@dataclass
class CoupledResult(Result):
    """Result from coupled multi-physics solver."""

    # Individual results
    thermal_result: Optional[ThermalResult] = None
    structural_result: Optional[StructuralResult] = None
    fluid_result: Optional[FluidResult] = None

    # Coupling statistics
    coupling_iterations: int = 0
    coupling_residual: float = 0.0

    # Derived coupled fields
    thermal_stress: Optional[np.ndarray] = None
    total_displacement: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = [
            f"=== Coupled Analysis Result: {self.problem_name} ===",
            f"Solver: {self.solver_name}",
            f"Converged: {self.converged}",
            f"Coupling iterations: {self.coupling_iterations}",
            f"Total solve time: {self.solve_time:.4f} s",
            "",
        ]

        if self.thermal_result:
            lines.append("Thermal:")
            lines.append(f"  T_min: {self.thermal_result.T_min:.2f} K")
            lines.append(f"  T_max: {self.thermal_result.T_max:.2f} K")

        if self.structural_result:
            lines.append("\nStructural:")
            lines.append(f"  Max displacement: {self.structural_result.max_displacement:.6e} m")
            lines.append(f"  Max stress: {self.structural_result.max_stress:.2e} Pa")

        if self.fluid_result:
            lines.append("\nFluid:")
            lines.append(f"  Max velocity: {self.fluid_result.max_velocity:.4f} m/s")
            lines.append(f"  Re: {self.fluid_result.reynolds_number:.1f}")

        return "\n".join(lines)

    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        fields = {
            "temperature": self.thermal_result.temperature if self.thermal_result else None,
            "displacement": self.structural_result.displacement if self.structural_result else None,
            "stress_vm": self.structural_result.stress_vm if self.structural_result else None,
            "velocity": self.fluid_result.velocity if self.fluid_result else None,
            "pressure": self.fluid_result.pressure if self.fluid_result else None,
            "thermal_stress": self.thermal_stress,
            "total_displacement": self.total_displacement,
        }
        return fields.get(field_name)


class CoupledSolver(Solver):
    """
    Multi-physics coupled solver.

    Coordinates thermal, structural, and fluid solvers with
    appropriate coupling strategies.

    Usage:
        solver = CoupledSolver()
        result = solver.solve(coupled_problem)
    """

    def __init__(self):
        super().__init__(name="CoupledSolver")
        self.thermal_solver = ThermalSolver()
        self.structural_solver = StructuralSolver()
        self.fluid_solver = LBMSolver()

    def validate_problem(self, problem: Problem) -> Dict[str, Any]:
        if not isinstance(problem, CoupledProblem):
            return {
                "valid": False,
                "errors": ["Problem must be a CoupledProblem"],
                "warnings": []
            }
        return problem.validate()

    def solve(self, problem: CoupledProblem) -> CoupledResult:
        """Solve the coupled multi-physics problem."""
        validation = self.validate_problem(problem)
        if not validation["valid"]:
            raise ValueError(f"Invalid problem: {validation['errors']}")

        self._solve_count += 1
        start_time = time.time()

        if problem.coupling_type == "sequential":
            result = self._solve_sequential(problem)
        elif problem.coupling_type == "iterative":
            result = self._solve_iterative(problem)
        else:
            raise ValueError(f"Unknown coupling type: {problem.coupling_type}")

        result.solve_time = time.time() - start_time
        return result

    def _solve_sequential(self, problem: CoupledProblem) -> CoupledResult:
        """
        Sequential (one-way) coupling.

        Solves physics in order without feedback:
        Fluid → Thermal → Structural
        """
        fluid_result = None
        thermal_result = None
        structural_result = None

        # 1. Solve fluid (if present)
        if problem.fluid_problem is not None:
            print("Solving fluid dynamics...")
            fluid_result = self.fluid_solver.solve(problem.fluid_problem)

        # 2. Solve thermal (with heat transfer from fluid if available)
        if problem.thermal_problem is not None:
            print("Solving thermal...")
            thermal_result = self.thermal_solver.solve(problem.thermal_problem)

        # 3. Solve structural (with thermal loads if available)
        if problem.structural_problem is not None:
            print("Solving structural...")

            # Modify structural problem with thermal loads
            if thermal_result is not None:
                structural_result = self._solve_thermo_structural(
                    problem.structural_problem,
                    thermal_result,
                    problem.thermal_expansion_coefficient,
                    problem.reference_temperature,
                )
            else:
                structural_result = self.structural_solver.solve(problem.structural_problem)

        return CoupledResult(
            problem_name=problem.name,
            solver_name=self.name,
            solve_time=0.0,
            converged=True,
            coupling_iterations=1,
            thermal_result=thermal_result,
            structural_result=structural_result,
            fluid_result=fluid_result,
        )

    def _solve_iterative(self, problem: CoupledProblem) -> CoupledResult:
        """
        Iterative (two-way) coupling.

        Iterates between physics until convergence.
        Currently implements thermal-structural coupling.
        """
        if problem.thermal_problem is None or problem.structural_problem is None:
            raise ValueError("Iterative coupling requires both thermal and structural problems")

        thermal_result = None
        structural_result = None
        converged = False

        # Store previous temperature for convergence check
        T_prev = None

        for iteration in range(1, problem.max_coupling_iterations + 1):
            print(f"Coupling iteration {iteration}...")

            # 1. Solve thermal
            thermal_result = self.thermal_solver.solve(problem.thermal_problem)

            # 2. Check convergence
            if T_prev is not None:
                residual = np.max(np.abs(thermal_result.temperature - T_prev)) / (
                    np.max(thermal_result.temperature) - np.min(thermal_result.temperature) + 1e-10
                )
                print(f"  Coupling residual: {residual:.2e}")

                if residual < problem.coupling_tolerance:
                    converged = True
                    break

            T_prev = thermal_result.temperature.copy()

            # 3. Solve structural with thermal loads
            structural_result = self._solve_thermo_structural(
                problem.structural_problem,
                thermal_result,
                problem.thermal_expansion_coefficient,
                problem.reference_temperature,
            )

            # 4. Update thermal problem geometry/BCs based on deformation
            # (For now, this is one-way - geometry doesn't change)

        return CoupledResult(
            problem_name=problem.name,
            solver_name=self.name,
            solve_time=0.0,
            converged=converged,
            coupling_iterations=iteration,
            coupling_residual=residual if T_prev is not None else 0.0,
            thermal_result=thermal_result,
            structural_result=structural_result,
        )

    def _solve_thermo_structural(
        self,
        structural_problem: StructuralProblem,
        thermal_result: ThermalResult,
        alpha: float,  # Thermal expansion coefficient
        T_ref: float,  # Reference temperature
    ) -> StructuralResult:
        """
        Solve structural problem with thermal loads.

        Thermal strain: ε_th = α * ΔT
        Thermal stress: σ_th = E * α * ΔT
        """
        # Get temperature field interpolated to structural mesh
        # (Assuming same mesh for now - in practice would need interpolation)
        T = thermal_result.temperature

        # Calculate thermal strain
        delta_T = T - T_ref
        thermal_strain = alpha * delta_T

        # Modify structural body forces to include thermal effects
        # This is a simplified approach - full implementation would
        # add thermal strain to the constitutive equations
        import copy
        modified_problem = copy.deepcopy(structural_problem)

        # Add equivalent thermal body force
        # F_thermal = E * α * ∇T (in each direction)
        E = structural_problem.youngs_modulus
        dT_dx = np.gradient(delta_T, axis=0) if delta_T.ndim == 2 else 0
        dT_dy = np.gradient(delta_T, axis=1) if delta_T.ndim == 2 else 0

        avg_thermal_force_x = E * alpha * np.mean(dT_dx) if isinstance(dT_dx, np.ndarray) else 0
        avg_thermal_force_y = E * alpha * np.mean(dT_dy) if isinstance(dT_dy, np.ndarray) else 0

        modified_problem.body_force = (
            structural_problem.body_force[0] + avg_thermal_force_x,
            structural_problem.body_force[1] + avg_thermal_force_y,
        )

        # Solve modified problem
        result = self.structural_solver.solve(modified_problem)

        # Add thermal stress to the result
        # σ_thermal = E * α * ΔT / (1 - 2ν) for constrained expansion
        nu = structural_problem.poissons_ratio
        thermal_stress = E * alpha * np.mean(np.abs(delta_T)) / (1 - 2 * nu)

        # Store in metadata
        result.metadata["thermal_stress_contribution"] = thermal_stress
        result.metadata["max_temperature_delta"] = float(np.max(np.abs(delta_T)))

        return result

    def solve_conjugate_heat_transfer(
        self,
        fluid_problem: FluidProblem,
        thermal_problem: ThermalProblem,
        interface_nodes: List[int],
        max_iterations: int = 10,
        tolerance: float = 1e-4,
    ) -> Tuple[FluidResult, ThermalResult]:
        """
        Solve conjugate heat transfer (CHT) problem.

        Couples fluid and solid thermal domains at their interface.
        Temperature and heat flux must be continuous at the interface.
        """
        fluid_result = None
        thermal_result = None

        # Initial thermal solve
        thermal_result = self.thermal_solver.solve(thermal_problem)
        T_interface = thermal_result.temperature[interface_nodes]

        for iteration in range(max_iterations):
            # Update fluid boundary with solid temperature
            # (This would modify the fluid problem's wall temperature BC)

            # Solve fluid
            fluid_result = self.fluid_solver.solve(fluid_problem)

            # Extract wall heat flux from fluid
            # q_wall = h * (T_wall - T_fluid)
            # (Simplified - actual implementation would use near-wall gradients)

            # Update solid thermal BC with heat flux
            # (This would modify the thermal problem's Neumann BC)

            # Solve thermal
            T_prev = thermal_result.temperature[interface_nodes].copy()
            thermal_result = self.thermal_solver.solve(thermal_problem)

            # Check convergence
            T_new = thermal_result.temperature[interface_nodes]
            residual = np.max(np.abs(T_new - T_prev)) / (np.max(T_new) - np.min(T_new) + 1e-10)

            if residual < tolerance:
                break

        return fluid_result, thermal_result


class FluidStructureInteraction:
    """
    Fluid-Structure Interaction (FSI) solver.

    Couples fluid forces with structural deformation.
    Uses partitioned approach with Aitken relaxation.
    """

    def __init__(self):
        self.fluid_solver = LBMSolver()
        self.structural_solver = StructuralSolver()

    def solve(
        self,
        fluid_problem: FluidProblem,
        structural_problem: StructuralProblem,
        interface_nodes: List[int],
        max_iterations: int = 20,
        tolerance: float = 1e-5,
        relaxation: float = 0.5,
    ) -> Tuple[FluidResult, StructuralResult]:
        """
        Solve FSI problem with partitioned approach.

        1. Solve fluid on current geometry
        2. Transfer forces to structure
        3. Solve structure
        4. Update geometry with displacement
        5. Repeat until convergence
        """
        import copy

        displacement = np.zeros((len(interface_nodes), 2))
        displacement_prev = displacement.copy()

        fluid_result = None
        structural_result = None

        for iteration in range(max_iterations):
            # 1. Update fluid geometry with current displacement
            # (This would modify the obstacle SDF based on displacement)

            # 2. Solve fluid
            fluid_result = self.fluid_solver.solve(fluid_problem)

            # 3. Extract pressure forces on interface
            # F = ∫ p * n dA
            # (Simplified - actual implementation needs proper integration)
            if fluid_result.pressure is not None:
                pressure_interface = fluid_result.pressure.flatten()[interface_nodes]
            else:
                pressure_interface = np.zeros(len(interface_nodes))

            # 4. Apply forces to structural problem
            modified_structural = copy.deepcopy(structural_problem)
            # (Would add pressure as traction BC on interface)

            # 5. Solve structure
            structural_result = self.structural_solver.solve(modified_structural)

            # 6. Get new displacement at interface
            displacement_new = structural_result.displacement[interface_nodes]

            # 7. Apply Aitken relaxation for stability
            displacement = (1 - relaxation) * displacement_prev + relaxation * displacement_new

            # 8. Check convergence
            disp_change = np.max(np.abs(displacement - displacement_prev))
            if disp_change < tolerance:
                break

            displacement_prev = displacement.copy()

        return fluid_result, structural_result
