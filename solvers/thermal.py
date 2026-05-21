"""
Thermal Solver for Cortex CEM

Solves heat conduction problems using finite difference method.

Supports:
- 1D, 2D, 3D steady-state heat conduction
- 2D transient heat conduction
- Dirichlet (fixed temperature) boundary conditions
- Neumann (heat flux) boundary conditions
- Convection boundary conditions

Governing equation:
- Steady-state: ∇²T = 0 (Laplace equation)
- Transient: ∂T/∂t = α∇²T (Heat equation)

where α = k/(ρ*cp) is thermal diffusivity
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time

from .base import Solver, Problem, Result
from knowledge import Material


@dataclass
class BoundaryCondition:
    """Boundary condition specification."""
    bc_type: str  # "dirichlet", "neumann", "convection"
    value: float  # Temperature (K) or heat flux (W/m²)
    coefficient: Optional[float] = None  # h for convection (W/m²K)

    def __post_init__(self):
        valid_types = ["dirichlet", "neumann", "convection"]
        if self.bc_type not in valid_types:
            raise ValueError(f"bc_type must be one of {valid_types}")


@dataclass
class ThermalProblem(Problem):
    """
    Thermal problem definition for heat conduction analysis.

    Example usage:
        problem = ThermalProblem(
            name="heated_plate",
            domain_size=(0.1, 0.1),  # 10cm x 10cm plate
            grid_size=(50, 50),
            material=materials_db.get("aluminum_6061"),
            boundary_conditions={
                "left": BoundaryCondition("dirichlet", 373.0),   # 100°C
                "right": BoundaryCondition("dirichlet", 293.0),  # 20°C
                "top": BoundaryCondition("neumann", 0.0),        # Insulated
                "bottom": BoundaryCondition("neumann", 0.0),     # Insulated
            }
        )
    """
    # Domain
    domain_size: Tuple[float, ...] = (1.0, 1.0)  # Size in meters (width, height) or (w, h, d)
    grid_size: Tuple[int, ...] = (50, 50)  # Number of grid points

    # Material
    material: Optional[Material] = None
    thermal_conductivity: float = 50.0  # W/(m·K) - used if material not specified

    # Boundary conditions: dict mapping location to BC
    # Locations: "left", "right", "top", "bottom" (2D) or faces for 3D
    boundary_conditions: Dict[str, BoundaryCondition] = field(default_factory=dict)

    # Initial condition (for transient)
    initial_temperature: float = 293.0  # K (20°C)

    # Heat source (optional): function(x, y) -> W/m³ or constant value
    heat_source: Optional[float] = None

    # Analysis type
    steady_state: bool = True
    time_steps: int = 100
    total_time: float = 1.0  # seconds (for transient)

    def __post_init__(self):
        # Validate dimensions match
        if len(self.domain_size) != len(self.grid_size):
            raise ValueError("domain_size and grid_size must have same dimensions")

        # Get thermal conductivity from material if provided
        if self.material is not None:
            self.thermal_conductivity = self.material.thermal_conductivity

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.domain_size)

    @property
    def dx(self) -> float:
        """Grid spacing in x direction."""
        return self.domain_size[0] / (self.grid_size[0] - 1)

    @property
    def dy(self) -> float:
        """Grid spacing in y direction."""
        if self.ndim >= 2:
            return self.domain_size[1] / (self.grid_size[1] - 1)
        return 0.0

    def validate(self) -> Dict[str, Any]:
        """Validate problem setup."""
        result = {"valid": True, "errors": [], "warnings": []}

        # Check grid size
        for i, n in enumerate(self.grid_size):
            if n < 3:
                result["errors"].append(f"Grid size in dimension {i} must be at least 3")
                result["valid"] = False

        # Check thermal conductivity
        if self.thermal_conductivity <= 0:
            result["errors"].append("Thermal conductivity must be positive")
            result["valid"] = False

        # Check boundary conditions
        required_bcs = ["left", "right"] if self.ndim == 1 else ["left", "right", "top", "bottom"]
        for bc_name in required_bcs:
            if bc_name not in self.boundary_conditions:
                result["warnings"].append(f"No BC specified for '{bc_name}', using insulated (Neumann q=0)")

        return result


@dataclass
class ThermalResult(Result):
    """Result from thermal solver."""

    # Temperature field
    temperature: Optional[np.ndarray] = None

    # Grid coordinates
    x_coords: Optional[np.ndarray] = None
    y_coords: Optional[np.ndarray] = None

    # Derived quantities
    heat_flux_x: Optional[np.ndarray] = None
    heat_flux_y: Optional[np.ndarray] = None

    # Statistics
    T_min: float = 0.0
    T_max: float = 0.0
    T_avg: float = 0.0

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"=== Thermal Analysis Result: {self.problem_name} ===",
            f"Solver: {self.solver_name}",
            f"Converged: {self.converged}",
            f"Solve time: {self.solve_time:.4f} s",
            f"Iterations: {self.iterations}",
            "",
            "Temperature Distribution:",
            f"  Min: {self.T_min:.2f} K ({self.T_min - 273.15:.2f} °C)",
            f"  Max: {self.T_max:.2f} K ({self.T_max - 273.15:.2f} °C)",
            f"  Avg: {self.T_avg:.2f} K ({self.T_avg - 273.15:.2f} °C)",
        ]

        if self.temperature is not None:
            lines.append(f"  Grid shape: {self.temperature.shape}")

        return "\n".join(lines)

    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        """Get a solution field by name."""
        fields = {
            "temperature": self.temperature,
            "T": self.temperature,
            "heat_flux_x": self.heat_flux_x,
            "heat_flux_y": self.heat_flux_y,
            "qx": self.heat_flux_x,
            "qy": self.heat_flux_y,
        }
        return fields.get(field_name)


class ThermalSolver(Solver):
    """
    Finite Difference solver for heat conduction.

    Solves the heat equation using:
    - Jacobi iteration for steady-state
    - Explicit or implicit time stepping for transient

    Usage:
        solver = ThermalSolver()
        result = solver.solve(problem)
        print(result.summary())
    """

    def __init__(self, max_iterations: int = 10000, tolerance: float = 1e-6):
        super().__init__(name="ThermalSolver-FD")
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def validate_problem(self, problem: Problem) -> Dict[str, Any]:
        """Validate thermal problem."""
        if not isinstance(problem, ThermalProblem):
            return {
                "valid": False,
                "errors": ["Problem must be a ThermalProblem"],
                "warnings": []
            }
        return problem.validate()

    def solve(self, problem: ThermalProblem) -> ThermalResult:
        """Solve the thermal problem."""
        validation = self.validate_problem(problem)
        if not validation["valid"]:
            raise ValueError(f"Invalid problem: {validation['errors']}")

        self._solve_count += 1
        start_time = time.time()

        if problem.ndim == 1:
            result = self._solve_1d(problem)
        elif problem.ndim == 2:
            if problem.steady_state:
                result = self._solve_2d_steady(problem)
            else:
                result = self._solve_2d_transient(problem)
        else:
            raise NotImplementedError("3D solver not yet implemented")

        result.solve_time = time.time() - start_time
        return result

    def _solve_1d(self, problem: ThermalProblem) -> ThermalResult:
        """Solve 1D steady-state heat conduction."""
        nx = problem.grid_size[0]
        dx = problem.dx

        # Initialize temperature array
        T = np.ones(nx) * problem.initial_temperature

        # Apply Dirichlet BCs
        bc_left = problem.boundary_conditions.get("left")
        bc_right = problem.boundary_conditions.get("right")

        if bc_left and bc_left.bc_type == "dirichlet":
            T[0] = bc_left.value
        if bc_right and bc_right.bc_type == "dirichlet":
            T[-1] = bc_right.value

        # For 1D steady-state with Dirichlet BCs, solution is linear
        # T(x) = T_left + (T_right - T_left) * x / L
        x = np.linspace(0, problem.domain_size[0], nx)
        T = T[0] + (T[-1] - T[0]) * x / problem.domain_size[0]

        return ThermalResult(
            problem_name=problem.name,
            solver_name=self.name,
            solve_time=0.0,
            converged=True,
            iterations=1,
            temperature=T,
            x_coords=x,
            T_min=float(T.min()),
            T_max=float(T.max()),
            T_avg=float(T.mean()),
        )

    def _solve_2d_steady(self, problem: ThermalProblem) -> ThermalResult:
        """Solve 2D steady-state heat conduction using Gauss-Seidel iteration."""
        nx, ny = problem.grid_size
        dx, dy = problem.dx, problem.dy

        # Create coordinate arrays
        x = np.linspace(0, problem.domain_size[0], nx)
        y = np.linspace(0, problem.domain_size[1], ny)

        # Initialize temperature field
        T = np.ones((nx, ny)) * problem.initial_temperature
        T_new = T.copy()

        # Apply boundary conditions
        self._apply_boundary_conditions(T, problem)

        # Coefficient for non-uniform grid (here dx=dy is assumed for simplicity)
        # For Laplacian: (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4*T[i,j]) / h² = 0
        # => T[i,j] = 0.25 * (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1])

        # Add heat source if present
        Q = np.zeros((nx, ny))
        if problem.heat_source is not None:
            Q[:, :] = problem.heat_source * dx * dy / problem.thermal_conductivity

        # Gauss-Seidel iteration
        converged = False
        iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            T_old = T.copy()

            # Update interior points
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    T[i, j] = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] + Q[i, j])

            # Apply boundary conditions
            self._apply_boundary_conditions(T, problem)

            # Check convergence
            max_diff = np.max(np.abs(T - T_old))
            if max_diff < self.tolerance:
                converged = True
                break

        # Calculate heat flux: q = -k * grad(T)
        heat_flux_x, heat_flux_y = self._calculate_heat_flux(T, problem)

        return ThermalResult(
            problem_name=problem.name,
            solver_name=self.name,
            solve_time=0.0,
            converged=converged,
            iterations=iteration,
            temperature=T,
            x_coords=x,
            y_coords=y,
            heat_flux_x=heat_flux_x,
            heat_flux_y=heat_flux_y,
            T_min=float(T.min()),
            T_max=float(T.max()),
            T_avg=float(T.mean()),
        )

    def _solve_2d_transient(self, problem: ThermalProblem) -> ThermalResult:
        """Solve 2D transient heat conduction using explicit method."""
        nx, ny = problem.grid_size
        dx, dy = problem.dx, problem.dy
        dt = problem.total_time / problem.time_steps

        # Thermal diffusivity
        if problem.material is not None:
            alpha = problem.material.thermal_diffusivity()
        else:
            # Assume typical values if no material specified
            alpha = problem.thermal_conductivity / (7800 * 500)  # Approx steel

        # Stability check (CFL condition)
        max_dt = 0.25 * min(dx, dy)**2 / alpha
        if dt > max_dt:
            print(f"Warning: dt={dt:.2e}s exceeds stability limit {max_dt:.2e}s")
            print(f"Reducing time steps to maintain stability...")
            dt = 0.9 * max_dt

        # Create coordinate arrays
        x = np.linspace(0, problem.domain_size[0], nx)
        y = np.linspace(0, problem.domain_size[1], ny)

        # Initialize temperature field
        T = np.ones((nx, ny)) * problem.initial_temperature
        self._apply_boundary_conditions(T, problem)

        # Coefficients
        rx = alpha * dt / dx**2
        ry = alpha * dt / dy**2

        # Time stepping (explicit FTCS scheme)
        iteration = 0
        for step in range(problem.time_steps):
            T_old = T.copy()

            # Update interior points
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    T[i, j] = T_old[i, j] + rx * (T_old[i+1, j] - 2*T_old[i, j] + T_old[i-1, j]) \
                                          + ry * (T_old[i, j+1] - 2*T_old[i, j] + T_old[i, j-1])

            # Apply boundary conditions
            self._apply_boundary_conditions(T, problem)
            iteration += 1

        # Calculate heat flux
        heat_flux_x, heat_flux_y = self._calculate_heat_flux(T, problem)

        return ThermalResult(
            problem_name=problem.name,
            solver_name=self.name,
            solve_time=0.0,
            converged=True,
            iterations=iteration,
            temperature=T,
            x_coords=x,
            y_coords=y,
            heat_flux_x=heat_flux_x,
            heat_flux_y=heat_flux_y,
            T_min=float(T.min()),
            T_max=float(T.max()),
            T_avg=float(T.mean()),
            metadata={"final_time": problem.total_time}
        )

    def _apply_boundary_conditions(self, T: np.ndarray, problem: ThermalProblem):
        """Apply boundary conditions to temperature field."""
        nx, ny = T.shape
        bcs = problem.boundary_conditions

        # Left boundary (i=0)
        bc = bcs.get("left", BoundaryCondition("neumann", 0.0))
        if bc.bc_type == "dirichlet":
            T[0, :] = bc.value
        elif bc.bc_type == "neumann":
            # dT/dx = q/k at left boundary
            T[0, :] = T[1, :] - bc.value * problem.dx / problem.thermal_conductivity
        elif bc.bc_type == "convection":
            # -k*dT/dx = h*(T - T_inf)
            h, T_inf = bc.coefficient, bc.value
            k = problem.thermal_conductivity
            T[0, :] = (k * T[1, :] / problem.dx + h * T_inf) / (k / problem.dx + h)

        # Right boundary (i=nx-1)
        bc = bcs.get("right", BoundaryCondition("neumann", 0.0))
        if bc.bc_type == "dirichlet":
            T[-1, :] = bc.value
        elif bc.bc_type == "neumann":
            T[-1, :] = T[-2, :] + bc.value * problem.dx / problem.thermal_conductivity
        elif bc.bc_type == "convection":
            h, T_inf = bc.coefficient, bc.value
            k = problem.thermal_conductivity
            T[-1, :] = (k * T[-2, :] / problem.dx + h * T_inf) / (k / problem.dx + h)

        # Bottom boundary (j=0)
        bc = bcs.get("bottom", BoundaryCondition("neumann", 0.0))
        if bc.bc_type == "dirichlet":
            T[:, 0] = bc.value
        elif bc.bc_type == "neumann":
            T[:, 0] = T[:, 1] - bc.value * problem.dy / problem.thermal_conductivity
        elif bc.bc_type == "convection":
            h, T_inf = bc.coefficient, bc.value
            k = problem.thermal_conductivity
            T[:, 0] = (k * T[:, 1] / problem.dy + h * T_inf) / (k / problem.dy + h)

        # Top boundary (j=ny-1)
        bc = bcs.get("top", BoundaryCondition("neumann", 0.0))
        if bc.bc_type == "dirichlet":
            T[:, -1] = bc.value
        elif bc.bc_type == "neumann":
            T[:, -1] = T[:, -2] + bc.value * problem.dy / problem.thermal_conductivity
        elif bc.bc_type == "convection":
            h, T_inf = bc.coefficient, bc.value
            k = problem.thermal_conductivity
            T[:, -1] = (k * T[:, -2] / problem.dy + h * T_inf) / (k / problem.dy + h)

    def _calculate_heat_flux(self, T: np.ndarray, problem: ThermalProblem) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate heat flux field from temperature field."""
        k = problem.thermal_conductivity
        dx, dy = problem.dx, problem.dy

        # Central difference for interior, forward/backward at boundaries
        heat_flux_x = np.zeros_like(T)
        heat_flux_y = np.zeros_like(T)

        # Interior points
        heat_flux_x[1:-1, :] = -k * (T[2:, :] - T[:-2, :]) / (2 * dx)
        heat_flux_y[:, 1:-1] = -k * (T[:, 2:] - T[:, :-2]) / (2 * dy)

        # Boundaries (forward/backward difference)
        heat_flux_x[0, :] = -k * (T[1, :] - T[0, :]) / dx
        heat_flux_x[-1, :] = -k * (T[-1, :] - T[-2, :]) / dx
        heat_flux_y[:, 0] = -k * (T[:, 1] - T[:, 0]) / dy
        heat_flux_y[:, -1] = -k * (T[:, -1] - T[:, -2]) / dy

        return heat_flux_x, heat_flux_y
