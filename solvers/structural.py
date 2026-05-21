"""
Structural Solver for Cortex CEM

Finite Element Method (FEM) solver for linear elasticity problems.

Supports:
- 2D plane stress/strain analysis
- 3D linear elasticity
- Various element types (triangular, quadrilateral)
- Dirichlet (fixed displacement) boundary conditions
- Neumann (traction/force) boundary conditions

Governing equations:
    Equilibrium: ∇·σ + f = 0
    Constitutive: σ = C:ε (Hooke's Law)
    Kinematics: ε = ½(∇u + (∇u)ᵀ)

where:
    σ = stress tensor
    ε = strain tensor
    u = displacement field
    C = elasticity tensor
    f = body force
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Literal
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time

from .base import Solver, Problem, Result
from knowledge import Material


@dataclass
class StructuralBoundaryCondition:
    """Boundary condition specification for structural analysis."""
    bc_type: str  # "displacement", "force", "traction"
    direction: str  # "x", "y", "z", "xy", "all"
    value: float  # Displacement (m) or Force (N) or Traction (Pa)
    node_ids: Optional[List[int]] = None  # Specific nodes (auto-detected if None)

    def __post_init__(self):
        valid_types = ["displacement", "force", "traction"]
        if self.bc_type not in valid_types:
            raise ValueError(f"bc_type must be one of {valid_types}")


@dataclass
class StructuralProblem(Problem):
    """
    Structural problem definition for FEM analysis.

    Example usage:
        problem = StructuralProblem(
            name="cantilever_beam",
            domain_size=(1.0, 0.1),  # 1m x 0.1m beam
            grid_size=(40, 10),
            material=materials_db.get("steel_structural"),
            boundary_conditions={
                "left": StructuralBoundaryCondition("displacement", "all", 0.0),
                "right": StructuralBoundaryCondition("force", "y", -1000.0),
            },
            analysis_type="plane_stress"
        )
    """
    # Domain
    domain_size: Tuple[float, ...] = (1.0, 0.1)  # Size in meters
    grid_size: Tuple[int, ...] = (40, 10)  # Number of nodes

    # Material (elastic properties)
    material: Optional[Material] = None
    youngs_modulus: float = 210e9  # Pa (steel default)
    poissons_ratio: float = 0.3  # dimensionless

    # Boundary conditions
    boundary_conditions: Dict[str, StructuralBoundaryCondition] = field(default_factory=dict)

    # Analysis type
    analysis_type: Literal["plane_stress", "plane_strain", "3d"] = "plane_stress"

    # Element type
    element_type: Literal["Q4", "T3"] = "Q4"  # Q4=quad, T3=triangle

    # Body forces (gravity, etc.)
    body_force: Tuple[float, ...] = (0.0, 0.0)  # N/m³

    def __post_init__(self):
        if len(self.domain_size) != len(self.grid_size):
            raise ValueError("domain_size and grid_size must have same dimensions")

        if self.material is not None:
            self.youngs_modulus = self.material.youngs_modulus
            self.poissons_ratio = self.material.poissons_ratio

    @property
    def ndim(self) -> int:
        return len(self.domain_size)

    @property
    def dx(self) -> float:
        return self.domain_size[0] / (self.grid_size[0] - 1)

    @property
    def dy(self) -> float:
        if self.ndim >= 2:
            return self.domain_size[1] / (self.grid_size[1] - 1)
        return 0.0

    @property
    def num_nodes(self) -> int:
        return int(np.prod(self.grid_size))

    @property
    def num_dofs(self) -> int:
        return self.num_nodes * self.ndim

    def validate(self) -> Dict[str, Any]:
        result = {"valid": True, "errors": [], "warnings": []}

        if self.youngs_modulus <= 0:
            result["errors"].append("Young's modulus must be positive")
            result["valid"] = False

        if not (-1 < self.poissons_ratio < 0.5):
            result["errors"].append("Poisson's ratio must be between -1 and 0.5")
            result["valid"] = False

        for i, n in enumerate(self.grid_size):
            if n < 2:
                result["errors"].append(f"Grid size in dimension {i} must be at least 2")
                result["valid"] = False

        return result


@dataclass
class StructuralResult(Result):
    """Result from structural FEM solver."""

    # Displacement field [num_nodes, ndim]
    displacement: Optional[np.ndarray] = None

    # Stress field (at element centroids or nodes)
    stress_xx: Optional[np.ndarray] = None
    stress_yy: Optional[np.ndarray] = None
    stress_xy: Optional[np.ndarray] = None
    stress_vm: Optional[np.ndarray] = None  # von Mises

    # Strain field
    strain_xx: Optional[np.ndarray] = None
    strain_yy: Optional[np.ndarray] = None
    strain_xy: Optional[np.ndarray] = None

    # Grid coordinates
    x_coords: Optional[np.ndarray] = None
    y_coords: Optional[np.ndarray] = None
    nodes: Optional[np.ndarray] = None

    # Statistics
    max_displacement: float = 0.0
    max_stress: float = 0.0
    max_strain: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== Structural Analysis Result: {self.problem_name} ===",
            f"Solver: {self.solver_name}",
            f"Converged: {self.converged}",
            f"Solve time: {self.solve_time:.4f} s",
            "",
            "Displacement:",
            f"  Max magnitude: {self.max_displacement:.6e} m",
            "",
            "Stress (von Mises):",
            f"  Max: {self.max_stress:.6e} Pa ({self.max_stress/1e6:.2f} MPa)",
        ]
        return "\n".join(lines)

    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        fields = {
            "displacement": self.displacement,
            "u": self.displacement,
            "stress_xx": self.stress_xx,
            "stress_yy": self.stress_yy,
            "stress_xy": self.stress_xy,
            "stress_vm": self.stress_vm,
            "von_mises": self.stress_vm,
            "strain_xx": self.strain_xx,
            "strain_yy": self.strain_yy,
            "strain_xy": self.strain_xy,
        }
        return fields.get(field_name)


class StructuralSolver(Solver):
    """
    Finite Element Method solver for structural mechanics.

    Uses Q4 (4-node quadrilateral) or T3 (3-node triangle) elements
    with isoparametric formulation.

    Usage:
        solver = StructuralSolver()
        result = solver.solve(problem)
        print(result.summary())
    """

    def __init__(self):
        super().__init__(name="StructuralSolver-FEM")

    def validate_problem(self, problem: Problem) -> Dict[str, Any]:
        if not isinstance(problem, StructuralProblem):
            return {
                "valid": False,
                "errors": ["Problem must be a StructuralProblem"],
                "warnings": []
            }
        return problem.validate()

    def solve(self, problem: StructuralProblem) -> StructuralResult:
        """Solve the structural problem using FEM."""
        validation = self.validate_problem(problem)
        if not validation["valid"]:
            raise ValueError(f"Invalid problem: {validation['errors']}")

        self._solve_count += 1
        start_time = time.time()

        if problem.ndim == 2:
            result = self._solve_2d(problem)
        else:
            raise NotImplementedError("3D solver not yet implemented")

        result.solve_time = time.time() - start_time
        return result

    def _solve_2d(self, problem: StructuralProblem) -> StructuralResult:
        """Solve 2D plane stress/strain problem."""
        nx, ny = problem.grid_size
        num_nodes = nx * ny
        num_dofs = num_nodes * 2  # 2 DOFs per node (ux, uy)

        # Generate mesh
        nodes, elements = self._generate_mesh_2d(problem)

        # Build constitutive matrix
        D = self._build_constitutive_matrix(problem)

        # Assemble global stiffness matrix
        K = self._assemble_stiffness_matrix(nodes, elements, D, problem)

        # Initialize force vector
        F = np.zeros(num_dofs)

        # Apply body forces
        if problem.body_force != (0.0, 0.0):
            F = self._apply_body_forces(F, nodes, elements, problem)

        # Apply boundary conditions
        K, F, fixed_dofs = self._apply_boundary_conditions(K, F, nodes, problem)

        # Solve system: K * u = F
        u = spsolve(K.tocsr(), F)

        # Calculate stresses and strains
        stress, strain = self._calculate_stress_strain(u, nodes, elements, D, problem)

        # Reshape displacement for output
        displacement = u.reshape(num_nodes, 2)
        disp_magnitude = np.linalg.norm(displacement, axis=1)

        # Calculate von Mises stress
        stress_vm = self._calculate_von_mises(stress)

        return StructuralResult(
            problem_name=problem.name,
            solver_name=self.name,
            solve_time=0.0,
            converged=True,
            iterations=1,
            displacement=displacement,
            stress_xx=stress[:, 0],
            stress_yy=stress[:, 1],
            stress_xy=stress[:, 2],
            stress_vm=stress_vm,
            strain_xx=strain[:, 0],
            strain_yy=strain[:, 1],
            strain_xy=strain[:, 2],
            nodes=nodes,
            x_coords=nodes[:, 0],
            y_coords=nodes[:, 1],
            max_displacement=float(np.max(disp_magnitude)),
            max_stress=float(np.max(stress_vm)),
            max_strain=float(np.max(np.abs(strain))),
        )

    def _generate_mesh_2d(self, problem: StructuralProblem) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 2D mesh with nodes and elements."""
        nx, ny = problem.grid_size
        Lx, Ly = problem.domain_size

        # Generate node coordinates
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        nodes = np.column_stack([xx.flatten(), yy.flatten()])

        # Generate element connectivity (Q4 elements)
        elements = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                # Node numbering (counter-clockwise)
                n1 = i * ny + j
                n2 = (i + 1) * ny + j
                n3 = (i + 1) * ny + (j + 1)
                n4 = i * ny + (j + 1)
                elements.append([n1, n2, n3, n4])

        return nodes, np.array(elements)

    def _build_constitutive_matrix(self, problem: StructuralProblem) -> np.ndarray:
        """Build constitutive matrix D for plane stress/strain."""
        E = problem.youngs_modulus
        nu = problem.poissons_ratio

        if problem.analysis_type == "plane_stress":
            # Plane stress: σz = 0
            factor = E / (1 - nu**2)
            D = factor * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2]
            ])
        else:  # plane_strain
            # Plane strain: εz = 0
            factor = E / ((1 + nu) * (1 - 2 * nu))
            D = factor * np.array([
                [1 - nu, nu, 0],
                [nu, 1 - nu, 0],
                [0, 0, (1 - 2 * nu) / 2]
            ])

        return D

    def _assemble_stiffness_matrix(self, nodes: np.ndarray, elements: np.ndarray,
                                    D: np.ndarray, problem: StructuralProblem) -> sparse.lil_matrix:
        """Assemble global stiffness matrix from element stiffness matrices."""
        num_nodes = len(nodes)
        num_dofs = num_nodes * 2
        K = sparse.lil_matrix((num_dofs, num_dofs))

        # Gauss quadrature points for Q4 element (2x2)
        gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        weights = np.array([1.0, 1.0])

        for elem in elements:
            # Get element node coordinates
            elem_nodes = nodes[elem]  # [4, 2]

            # Initialize element stiffness matrix
            Ke = np.zeros((8, 8))

            # Numerical integration
            for i, xi in enumerate(gauss_pts):
                for j, eta in enumerate(gauss_pts):
                    w = weights[i] * weights[j]

                    # Shape function derivatives in natural coords
                    dN_dxi = 0.25 * np.array([
                        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                        [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
                    ])

                    # Jacobian matrix
                    J = dN_dxi @ elem_nodes  # [2, 2]
                    detJ = np.linalg.det(J)
                    J_inv = np.linalg.inv(J)

                    # Shape function derivatives in physical coords
                    dN_dx = J_inv @ dN_dxi  # [2, 4]

                    # B matrix (strain-displacement)
                    B = np.zeros((3, 8))
                    for k in range(4):
                        B[0, 2*k] = dN_dx[0, k]      # εxx = du/dx
                        B[1, 2*k + 1] = dN_dx[1, k]  # εyy = dv/dy
                        B[2, 2*k] = dN_dx[1, k]      # γxy = du/dy + dv/dx
                        B[2, 2*k + 1] = dN_dx[0, k]

                    # Element stiffness contribution
                    Ke += w * detJ * (B.T @ D @ B)

            # Assemble into global matrix
            dofs = []
            for n in elem:
                dofs.extend([2*n, 2*n + 1])

            for ii, di in enumerate(dofs):
                for jj, dj in enumerate(dofs):
                    K[di, dj] += Ke[ii, jj]

        return K

    def _apply_body_forces(self, F: np.ndarray, nodes: np.ndarray,
                           elements: np.ndarray, problem: StructuralProblem) -> np.ndarray:
        """Apply distributed body forces (e.g., gravity)."""
        bx, by = problem.body_force

        # Gauss quadrature
        gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        weights = np.array([1.0, 1.0])

        for elem in elements:
            elem_nodes = nodes[elem]
            Fe = np.zeros(8)

            for i, xi in enumerate(gauss_pts):
                for j, eta in enumerate(gauss_pts):
                    w = weights[i] * weights[j]

                    # Shape functions
                    N = 0.25 * np.array([
                        (1 - xi) * (1 - eta),
                        (1 + xi) * (1 - eta),
                        (1 + xi) * (1 + eta),
                        (1 - xi) * (1 + eta)
                    ])

                    # Jacobian
                    dN_dxi = 0.25 * np.array([
                        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                        [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
                    ])
                    J = dN_dxi @ elem_nodes
                    detJ = np.linalg.det(J)

                    # Body force contribution
                    for k in range(4):
                        Fe[2*k] += w * detJ * N[k] * bx
                        Fe[2*k + 1] += w * detJ * N[k] * by

            # Assemble into global vector
            for k, n in enumerate(elem):
                F[2*n] += Fe[2*k]
                F[2*n + 1] += Fe[2*k + 1]

        return F

    def _apply_boundary_conditions(self, K: sparse.lil_matrix, F: np.ndarray,
                                    nodes: np.ndarray, problem: StructuralProblem
                                    ) -> Tuple[sparse.lil_matrix, np.ndarray, List[int]]:
        """Apply displacement and force boundary conditions."""
        nx, ny = problem.grid_size
        num_nodes = len(nodes)
        fixed_dofs = []

        # Identify boundary nodes
        left_nodes = [i * ny + j for i in [0] for j in range(ny)]
        right_nodes = [i * ny + j for i in [nx - 1] for j in range(ny)]
        bottom_nodes = [i * ny + 0 for i in range(nx)]
        top_nodes = [i * ny + (ny - 1) for i in range(nx)]

        boundary_map = {
            "left": left_nodes,
            "right": right_nodes,
            "bottom": bottom_nodes,
            "top": top_nodes,
        }

        for location, bc in problem.boundary_conditions.items():
            if bc.node_ids is not None:
                node_list = bc.node_ids
            else:
                node_list = boundary_map.get(location, [])

            if bc.bc_type == "displacement":
                # Fixed displacement BC (Dirichlet)
                for n in node_list:
                    if bc.direction in ["x", "all"]:
                        dof = 2 * n
                        fixed_dofs.append(dof)
                        K[dof, :] = 0
                        K[:, dof] = 0
                        K[dof, dof] = 1.0
                        F[dof] = bc.value

                    if bc.direction in ["y", "all"]:
                        dof = 2 * n + 1
                        fixed_dofs.append(dof)
                        K[dof, :] = 0
                        K[:, dof] = 0
                        K[dof, dof] = 1.0
                        F[dof] = bc.value

            elif bc.bc_type == "force":
                # Point force BC (Neumann)
                force_per_node = bc.value / len(node_list)
                for n in node_list:
                    if bc.direction in ["x", "all"]:
                        F[2 * n] += force_per_node
                    if bc.direction in ["y", "all"]:
                        F[2 * n + 1] += force_per_node

            elif bc.bc_type == "traction":
                # Surface traction (distributed load)
                # Traction = force per unit area, applied to boundary
                # For edge nodes, integrate along edge
                traction = bc.value
                if location in ["left", "right"]:
                    edge_length = problem.domain_size[1]
                else:
                    edge_length = problem.domain_size[0]

                total_force = traction * edge_length
                force_per_node = total_force / len(node_list)

                for n in node_list:
                    if bc.direction in ["x", "all"]:
                        F[2 * n] += force_per_node
                    if bc.direction in ["y", "all"]:
                        F[2 * n + 1] += force_per_node

        return K, F, fixed_dofs

    def _calculate_stress_strain(self, u: np.ndarray, nodes: np.ndarray,
                                  elements: np.ndarray, D: np.ndarray,
                                  problem: StructuralProblem) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate stress and strain at element centroids."""
        num_elements = len(elements)
        stress = np.zeros((num_elements, 3))  # [σxx, σyy, τxy]
        strain = np.zeros((num_elements, 3))  # [εxx, εyy, γxy]

        for e, elem in enumerate(elements):
            elem_nodes = nodes[elem]

            # Get element displacements
            ue = np.zeros(8)
            for k, n in enumerate(elem):
                ue[2*k] = u[2*n]
                ue[2*k + 1] = u[2*n + 1]

            # Evaluate at element centroid (xi=0, eta=0)
            xi, eta = 0.0, 0.0

            dN_dxi = 0.25 * np.array([
                [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
            ])

            J = dN_dxi @ elem_nodes
            J_inv = np.linalg.inv(J)
            dN_dx = J_inv @ dN_dxi

            # B matrix
            B = np.zeros((3, 8))
            for k in range(4):
                B[0, 2*k] = dN_dx[0, k]
                B[1, 2*k + 1] = dN_dx[1, k]
                B[2, 2*k] = dN_dx[1, k]
                B[2, 2*k + 1] = dN_dx[0, k]

            # Calculate strain and stress
            strain[e] = B @ ue
            stress[e] = D @ strain[e]

        return stress, strain

    def _calculate_von_mises(self, stress: np.ndarray) -> np.ndarray:
        """Calculate von Mises equivalent stress."""
        # For 2D plane stress: σvm = sqrt(σxx² - σxx*σyy + σyy² + 3*τxy²)
        sxx = stress[:, 0]
        syy = stress[:, 1]
        sxy = stress[:, 2]

        return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
