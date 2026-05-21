"""
CFD Solver for Cortex CEM - Lattice Boltzmann Method (LBM)

Implements the D2Q9 and D3Q19 Lattice Boltzmann schemes for
incompressible Navier-Stokes equations.

Supports:
- 2D/3D incompressible flow
- No-slip walls (bounce-back)
- Velocity inlet/outlet
- Pressure boundaries
- Obstacles via SDF geometry

The LBM solves:
    ∂f/∂t + c·∇f = Ω(f)

which recovers Navier-Stokes in the macroscopic limit:
    ∂ρ/∂t + ∇·(ρu) = 0           (continuity)
    ∂(ρu)/∂t + ∇·(ρuu) = -∇p + μ∇²u   (momentum)

Advantages:
- Natural parallelization
- Easy handling of complex geometries
- No mesh generation required
- Explicit time stepping
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable, Literal
import numpy as np
import time

from .base import Solver, Problem, Result


# D2Q9 lattice constants
D2Q9_WEIGHTS = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
D2Q9_VELOCITIES = np.array([
    [0, 0],   # 0: rest
    [1, 0],   # 1: east
    [0, 1],   # 2: north
    [-1, 0],  # 3: west
    [0, -1],  # 4: south
    [1, 1],   # 5: northeast
    [-1, 1],  # 6: northwest
    [-1, -1], # 7: southwest
    [1, -1],  # 8: southeast
])
D2Q9_OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# D3Q19 lattice constants
D3Q19_WEIGHTS = np.array([
    1/3,  # rest
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,  # faces
    1/36, 1/36, 1/36, 1/36, 1/36, 1/36,  # edges xy
    1/36, 1/36, 1/36, 1/36, 1/36, 1/36,  # edges xz, yz
])
D3Q19_VELOCITIES = np.array([
    [0, 0, 0],    # 0: rest
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],  # faces
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],  # edges in xy
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],  # edges in xz
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],  # edges in yz
])
D3Q19_OPPOSITE = np.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15])


@dataclass
class FluidBoundaryCondition:
    """Boundary condition for CFD."""
    bc_type: str  # "wall", "velocity", "pressure", "outflow"
    velocity: Tuple[float, ...] = (0.0, 0.0)  # For velocity BC
    pressure: float = 0.0  # For pressure BC (relative)

    def __post_init__(self):
        valid_types = ["wall", "velocity", "pressure", "outflow"]
        if self.bc_type not in valid_types:
            raise ValueError(f"bc_type must be one of {valid_types}")


@dataclass
class FluidProblem(Problem):
    """
    Fluid dynamics problem for LBM solver.

    Example:
        problem = FluidProblem(
            name="channel_flow",
            domain_size=(1.0, 0.2),
            grid_size=(200, 40),
            viscosity=1e-6,
            boundary_conditions={
                "left": FluidBoundaryCondition("velocity", velocity=(0.1, 0.0)),
                "right": FluidBoundaryCondition("outflow"),
                "top": FluidBoundaryCondition("wall"),
                "bottom": FluidBoundaryCondition("wall"),
            }
        )
    """
    # Domain
    domain_size: Tuple[float, ...] = (1.0, 0.2)  # meters
    grid_size: Tuple[int, ...] = (100, 20)  # lattice nodes

    # Fluid properties
    density: float = 1.0  # kg/m³ (reference)
    viscosity: float = 1e-6  # m²/s (kinematic viscosity)

    # LBM parameters (auto-calculated if not specified)
    tau: Optional[float] = None  # Relaxation time

    # Boundary conditions
    boundary_conditions: Dict[str, FluidBoundaryCondition] = field(default_factory=dict)

    # Obstacle (SDF function: negative inside solid)
    obstacle_sdf: Optional[Callable] = None

    # Simulation parameters
    max_iterations: int = 10000
    convergence_tol: float = 1e-6
    output_interval: int = 100

    # Characteristic velocity (for Re calculation)
    u_ref: float = 0.1  # m/s

    def __post_init__(self):
        # Calculate relaxation time from viscosity
        if self.tau is None:
            # In lattice units: ν = (τ - 0.5) * cs² * Δt
            # With cs² = 1/3 and Δt = 1: ν = (τ - 0.5) / 3
            # But we need to convert physical viscosity to lattice units
            dx = self.domain_size[0] / self.grid_size[0]
            dt = dx / self.u_ref  # Assuming Ma << 1
            nu_lattice = self.viscosity * dt / (dx * dx)
            self.tau = 3 * nu_lattice + 0.5

    @property
    def ndim(self) -> int:
        return len(self.domain_size)

    @property
    def reynolds_number(self) -> float:
        """Estimate Reynolds number based on domain size."""
        L = min(self.domain_size)
        return self.u_ref * L / self.viscosity

    def validate(self) -> Dict[str, Any]:
        result = {"valid": True, "errors": [], "warnings": []}

        if self.viscosity <= 0:
            result["errors"].append("Viscosity must be positive")
            result["valid"] = False

        if self.tau is not None and self.tau <= 0.5:
            result["errors"].append("Relaxation time tau must be > 0.5 for stability")
            result["valid"] = False

        if self.tau is not None and self.tau > 2.0:
            result["warnings"].append(f"tau={self.tau:.2f} is large, may be inaccurate")

        return result


@dataclass
class FluidResult(Result):
    """Result from LBM CFD solver."""

    # Velocity field [nx, ny, 2] or [nx, ny, nz, 3]
    velocity: Optional[np.ndarray] = None

    # Pressure field (actually density in LBM)
    pressure: Optional[np.ndarray] = None
    density: Optional[np.ndarray] = None

    # Derived quantities
    velocity_magnitude: Optional[np.ndarray] = None
    vorticity: Optional[np.ndarray] = None

    # Grid
    x_coords: Optional[np.ndarray] = None
    y_coords: Optional[np.ndarray] = None

    # Statistics
    max_velocity: float = 0.0
    avg_velocity: float = 0.0
    residual: float = 0.0
    reynolds_number: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== CFD Analysis Result: {self.problem_name} ===",
            f"Solver: {self.solver_name}",
            f"Converged: {self.converged}",
            f"Iterations: {self.iterations}",
            f"Solve time: {self.solve_time:.4f} s",
            f"Final residual: {self.residual:.2e}",
            "",
            f"Reynolds number: {self.reynolds_number:.1f}",
            f"Max velocity: {self.max_velocity:.4f} m/s",
            f"Avg velocity: {self.avg_velocity:.4f} m/s",
        ]
        return "\n".join(lines)

    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        fields = {
            "velocity": self.velocity,
            "u": self.velocity,
            "pressure": self.pressure,
            "p": self.pressure,
            "density": self.density,
            "rho": self.density,
            "velocity_magnitude": self.velocity_magnitude,
            "vorticity": self.vorticity,
        }
        return fields.get(field_name)


class LBMSolver(Solver):
    """
    Lattice Boltzmann Method solver for incompressible flow.

    Implements:
    - BGK collision operator with single relaxation time
    - Zou-He velocity/pressure boundary conditions
    - Bounce-back for walls
    - Obstacle handling via SDF

    Usage:
        solver = LBMSolver()
        result = solver.solve(problem)
        print(result.summary())
    """

    def __init__(self, collision_model: str = "bgk"):
        super().__init__(name="LBMSolver-D2Q9")
        self.collision_model = collision_model

    def validate_problem(self, problem: Problem) -> Dict[str, Any]:
        if not isinstance(problem, FluidProblem):
            return {
                "valid": False,
                "errors": ["Problem must be a FluidProblem"],
                "warnings": []
            }
        return problem.validate()

    def solve(self, problem: FluidProblem) -> FluidResult:
        """Solve the fluid dynamics problem."""
        validation = self.validate_problem(problem)
        if not validation["valid"]:
            raise ValueError(f"Invalid problem: {validation['errors']}")

        self._solve_count += 1
        start_time = time.time()

        if problem.ndim == 2:
            result = self._solve_2d(problem)
        else:
            raise NotImplementedError("3D LBM not yet implemented")

        result.solve_time = time.time() - start_time
        return result

    def _solve_2d(self, problem: FluidProblem) -> FluidResult:
        """Solve 2D flow using D2Q9 lattice."""
        nx, ny = problem.grid_size
        tau = problem.tau
        omega = 1.0 / tau

        # Lattice constants
        w = D2Q9_WEIGHTS
        c = D2Q9_VELOCITIES
        opp = D2Q9_OPPOSITE

        # Initialize distribution functions at equilibrium
        rho = np.ones((nx, ny)) * problem.density
        ux = np.zeros((nx, ny))
        uy = np.zeros((nx, ny))

        # Initialize with inlet velocity if specified
        if "left" in problem.boundary_conditions:
            bc = problem.boundary_conditions["left"]
            if bc.bc_type == "velocity":
                ux[:, :] = bc.velocity[0]
                if len(bc.velocity) > 1:
                    uy[:, :] = bc.velocity[1]

        f = self._equilibrium(rho, ux, uy)

        # Create obstacle mask
        obstacle = np.zeros((nx, ny), dtype=bool)
        if problem.obstacle_sdf is not None:
            x = np.linspace(0, problem.domain_size[0], nx)
            y = np.linspace(0, problem.domain_size[1], ny)
            xx, yy = np.meshgrid(x, y, indexing='ij')
            points = np.stack([xx, yy], axis=-1).reshape(-1, 2)
            sdf_values = problem.obstacle_sdf(points)
            obstacle = (sdf_values.reshape(nx, ny) < 0)

        # Identify boundary nodes
        left = np.zeros((nx, ny), dtype=bool)
        left[0, :] = True
        right = np.zeros((nx, ny), dtype=bool)
        right[-1, :] = True
        top = np.zeros((nx, ny), dtype=bool)
        top[:, -1] = True
        bottom = np.zeros((nx, ny), dtype=bool)
        bottom[:, 0] = True

        # Main iteration loop
        converged = False
        residual = 1.0

        for iteration in range(1, problem.max_iterations + 1):
            ux_old = ux.copy()
            uy_old = uy.copy()

            # 1. Collision step (BGK)
            feq = self._equilibrium(rho, ux, uy)
            f = f - omega * (f - feq)

            # 2. Streaming step
            f = self._stream(f, c)

            # 3. Boundary conditions
            f, rho, ux, uy = self._apply_boundaries(
                f, rho, ux, uy, problem, left, right, top, bottom
            )

            # 4. Obstacle bounce-back
            if problem.obstacle_sdf is not None:
                f = self._bounce_back(f, obstacle, opp)

            # 5. Compute macroscopic quantities
            rho, ux, uy = self._compute_macroscopic(f, c)

            # Apply obstacle mask
            ux[obstacle] = 0
            uy[obstacle] = 0

            # Check convergence
            if iteration % problem.output_interval == 0:
                du = np.sqrt((ux - ux_old)**2 + (uy - uy_old)**2)
                residual = np.max(du) / (np.max(np.sqrt(ux**2 + uy**2)) + 1e-10)

                if residual < problem.convergence_tol:
                    converged = True
                    break

        # Convert to physical units
        dx = problem.domain_size[0] / nx
        dt = dx / problem.u_ref

        velocity = np.stack([ux, uy], axis=-1) * (dx / dt)
        velocity_magnitude = np.sqrt(ux**2 + uy**2) * (dx / dt)

        # Calculate pressure from density (ideal gas EOS: p = ρ * cs²)
        cs2 = 1/3  # speed of sound squared in lattice units
        pressure = (rho - 1.0) * cs2 * (dx/dt)**2

        # Calculate vorticity: ω = ∂uy/∂x - ∂ux/∂y
        vorticity = np.zeros((nx, ny))
        vorticity[1:-1, 1:-1] = (
            (uy[2:, 1:-1] - uy[:-2, 1:-1]) / (2*dx) -
            (ux[1:-1, 2:] - ux[1:-1, :-2]) / (2*dx)
        ) * (dx / dt)

        # Grid coordinates
        x_coords = np.linspace(0, problem.domain_size[0], nx)
        y_coords = np.linspace(0, problem.domain_size[1], ny)

        return FluidResult(
            problem_name=problem.name,
            solver_name=self.name,
            solve_time=0.0,
            converged=converged,
            iterations=iteration,
            velocity=velocity,
            pressure=pressure,
            density=rho,
            velocity_magnitude=velocity_magnitude,
            vorticity=vorticity,
            x_coords=x_coords,
            y_coords=y_coords,
            max_velocity=float(np.max(velocity_magnitude)),
            avg_velocity=float(np.mean(velocity_magnitude[~obstacle])),
            residual=residual,
            reynolds_number=problem.reynolds_number,
        )

    def _equilibrium(self, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        """Calculate equilibrium distribution function."""
        w = D2Q9_WEIGHTS
        c = D2Q9_VELOCITIES

        nx, ny = rho.shape
        feq = np.zeros((9, nx, ny))

        usq = ux**2 + uy**2

        for i in range(9):
            cu = c[i, 0] * ux + c[i, 1] * uy
            feq[i] = w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)

        return feq

    def _stream(self, f: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Stream distribution functions to neighbors."""
        f_new = np.zeros_like(f)

        for i in range(9):
            f_new[i] = np.roll(np.roll(f[i], c[i, 0], axis=0), c[i, 1], axis=1)

        return f_new

    def _compute_macroscopic(self, f: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute density and velocity from distribution functions."""
        rho = np.sum(f, axis=0)
        ux = np.sum(f * c[:, 0, np.newaxis, np.newaxis], axis=0) / rho
        uy = np.sum(f * c[:, 1, np.newaxis, np.newaxis], axis=0) / rho
        return rho, ux, uy

    def _bounce_back(self, f: np.ndarray, obstacle: np.ndarray, opp: np.ndarray) -> np.ndarray:
        """Apply bounce-back at obstacle nodes."""
        f_temp = f.copy()
        for i in range(9):
            f[i][obstacle] = f_temp[opp[i]][obstacle]
        return f

    def _apply_boundaries(self, f: np.ndarray, rho: np.ndarray,
                          ux: np.ndarray, uy: np.ndarray,
                          problem: FluidProblem,
                          left: np.ndarray, right: np.ndarray,
                          top: np.ndarray, bottom: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply boundary conditions using Zou-He method."""
        w = D2Q9_WEIGHTS
        c = D2Q9_VELOCITIES
        opp = D2Q9_OPPOSITE

        for location, bc in problem.boundary_conditions.items():
            if bc.bc_type == "wall":
                # Bounce-back for walls
                if location == "left":
                    mask = left
                elif location == "right":
                    mask = right
                elif location == "top":
                    mask = top
                elif location == "bottom":
                    mask = bottom
                else:
                    continue

                f_temp = f.copy()
                for i in range(9):
                    f[i][mask] = f_temp[opp[i]][mask]

            elif bc.bc_type == "velocity":
                # Zou-He velocity boundary condition
                u_bc = bc.velocity[0]
                v_bc = bc.velocity[1] if len(bc.velocity) > 1 else 0.0

                if location == "left":
                    # Known: f3, f4, f6, f7 (coming from inside)
                    # Unknown: f1, f5, f8 (going to inside)
                    rho[0, :] = (f[0, 0, :] + f[2, 0, :] + f[4, 0, :] +
                                 2*(f[3, 0, :] + f[6, 0, :] + f[7, 0, :])) / (1 - u_bc)
                    ux[0, :] = u_bc
                    uy[0, :] = v_bc

                    f[1, 0, :] = f[3, 0, :] + (2/3) * rho[0, :] * u_bc
                    f[5, 0, :] = f[7, 0, :] - 0.5*(f[2, 0, :] - f[4, 0, :]) + (1/6)*rho[0, :]*u_bc + 0.5*rho[0, :]*v_bc
                    f[8, 0, :] = f[6, 0, :] + 0.5*(f[2, 0, :] - f[4, 0, :]) + (1/6)*rho[0, :]*u_bc - 0.5*rho[0, :]*v_bc

                elif location == "right":
                    # Outlet velocity BC (less common)
                    ux[-1, :] = u_bc
                    uy[-1, :] = v_bc
                    rho[-1, :] = (f[0, -1, :] + f[2, -1, :] + f[4, -1, :] +
                                  2*(f[1, -1, :] + f[5, -1, :] + f[8, -1, :])) / (1 + u_bc)

                    f[3, -1, :] = f[1, -1, :] - (2/3) * rho[-1, :] * u_bc
                    f[6, -1, :] = f[8, -1, :] - 0.5*(f[2, -1, :] - f[4, -1, :]) - (1/6)*rho[-1, :]*u_bc + 0.5*rho[-1, :]*v_bc
                    f[7, -1, :] = f[5, -1, :] + 0.5*(f[2, -1, :] - f[4, -1, :]) - (1/6)*rho[-1, :]*u_bc - 0.5*rho[-1, :]*v_bc

            elif bc.bc_type == "outflow":
                # Zero-gradient (copy from interior)
                if location == "right":
                    f[:, -1, :] = f[:, -2, :]
                    rho[-1, :] = rho[-2, :]
                    ux[-1, :] = ux[-2, :]
                    uy[-1, :] = uy[-2, :]
                elif location == "left":
                    f[:, 0, :] = f[:, 1, :]
                    rho[0, :] = rho[1, :]
                    ux[0, :] = ux[1, :]
                    uy[0, :] = uy[1, :]

            elif bc.bc_type == "pressure":
                # Zou-He pressure boundary (constant density)
                rho_bc = problem.density + bc.pressure / (1/3)  # p = rho * cs²

                if location == "right":
                    rho[-1, :] = rho_bc
                    u_bc = -1 + (f[0, -1, :] + f[2, -1, :] + f[4, -1, :] +
                                 2*(f[1, -1, :] + f[5, -1, :] + f[8, -1, :])) / rho_bc
                    ux[-1, :] = u_bc
                    uy[-1, :] = 0

                    f[3, -1, :] = f[1, -1, :] - (2/3) * rho_bc * u_bc
                    f[6, -1, :] = f[8, -1, :] - 0.5*(f[2, -1, :] - f[4, -1, :]) - (1/6)*rho_bc*u_bc
                    f[7, -1, :] = f[5, -1, :] + 0.5*(f[2, -1, :] - f[4, -1, :]) - (1/6)*rho_bc*u_bc

        return f, rho, ux, uy


class ChannelFlowBenchmark:
    """
    Benchmark for Poiseuille channel flow.

    Analytical solution exists for validation.
    """

    @staticmethod
    def analytical_velocity(y: np.ndarray, H: float, dp_dx: float, mu: float) -> np.ndarray:
        """
        Analytical velocity profile for Poiseuille flow.

        u(y) = (dp/dx) * y * (H - y) / (2 * mu)

        Args:
            y: y-coordinates (0 to H)
            H: Channel height
            dp_dx: Pressure gradient (negative for flow in +x)
            mu: Dynamic viscosity
        """
        return -dp_dx * y * (H - y) / (2 * mu)

    @staticmethod
    def max_velocity(H: float, dp_dx: float, mu: float) -> float:
        """Maximum velocity at channel centerline."""
        return -dp_dx * H**2 / (8 * mu)


class CylinderFlowBenchmark:
    """
    Benchmark for flow around a cylinder.

    Used to validate vortex shedding and drag calculations.
    """

    @staticmethod
    def strouhal_number(Re: float) -> float:
        """
        Empirical Strouhal number for vortex shedding.

        St ≈ 0.21 * (1 - 21.2/Re) for 40 < Re < 1000
        """
        if Re < 40:
            return 0.0  # No shedding
        return 0.21 * (1 - 21.2 / Re)

    @staticmethod
    def drag_coefficient(Re: float) -> float:
        """
        Empirical drag coefficient for cylinder.

        Various correlations for different Re ranges.
        """
        if Re < 1:
            return 24 / Re  # Stokes regime
        elif Re < 1000:
            return 24/Re * (1 + 0.15 * Re**0.687)  # Intermediate
        else:
            return 0.44  # Newton regime (approximate)
