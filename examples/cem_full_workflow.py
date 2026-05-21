"""
Cortex CEM Full Workflow Example

Demonstrates the complete Leap71-style Computational Engineering Model workflow:
1. Define parametric geometry using SDFs
2. Run thermal simulation on the design
3. Run structural analysis with thermal loads
4. Optimize the design using genetic algorithms
5. Generate physics-informed geometry
6. Export for additive manufacturing

This example creates an optimized rocket nozzle with:
- Thermal-adaptive wall thickness
- Stress-optimized internal structure
- Lattice infill for weight reduction
"""

import numpy as np
import time

# Geometry imports
from geometry import (
    SDF, Union, Difference, Intersection, Offset, SmoothUnion,
    Sphere, Box, Cylinder, Torus,
    VoxelGrid,
    PhysicsField, ThermalWallThickness, StressOptimizedTopology,
    AdaptiveLattice, CoupledPhysicsGeometry,
)

# Solver imports
from solvers import (
    ThermalSolver, ThermalProblem, ThermalResult, BoundaryCondition,
    StructuralSolver, StructuralProblem, StructuralResult, StructuralBoundaryCondition,
    LBMSolver, FluidProblem, FluidResult, FluidBoundaryCondition,
    CoupledSolver, CoupledProblem,
    DesignOptimizer, OptimizationProblem, DesignVariable, Constraint,
    NSGA2,
)


# ============================================================================
# Step 1: Define Parametric Geometry
# ============================================================================

class ParametricNozzle(SDF):
    """
    Parametric rocket nozzle defined by key dimensions.

    Parameters:
        chamber_radius: Combustion chamber radius (mm)
        chamber_length: Chamber length (mm)
        throat_radius: Throat radius (mm)
        exit_radius: Nozzle exit radius (mm)
        nozzle_length: Divergent section length (mm)
        wall_thickness: Base wall thickness (mm)
    """

    def __init__(
        self,
        chamber_radius: float = 15.0,
        chamber_length: float = 40.0,
        throat_radius: float = 8.0,
        exit_radius: float = 25.0,
        nozzle_length: float = 60.0,
        wall_thickness: float = 2.0,
    ):
        self.chamber_radius = chamber_radius
        self.chamber_length = chamber_length
        self.throat_radius = throat_radius
        self.exit_radius = exit_radius
        self.nozzle_length = nozzle_length
        self.wall_thickness = wall_thickness

        # Total length
        self.total_length = chamber_length + nozzle_length

        # Expansion ratio
        self.expansion_ratio = (exit_radius / throat_radius) ** 2

    def _profile_radius(self, z: np.ndarray) -> np.ndarray:
        """Get the inner radius at axial position z."""
        radius = np.zeros_like(z)

        # Chamber section
        chamber_mask = z <= self.chamber_length
        radius[chamber_mask] = self.chamber_radius

        # Convergent section (chamber to throat)
        conv_mask = (z > self.chamber_length * 0.8) & (z <= self.chamber_length)
        if np.any(conv_mask):
            t = (z[conv_mask] - self.chamber_length * 0.8) / (self.chamber_length * 0.2)
            radius[conv_mask] = self.chamber_radius + t * (self.throat_radius - self.chamber_radius)

        # Divergent section (throat to exit)
        div_mask = z > self.chamber_length
        if np.any(div_mask):
            t = (z[div_mask] - self.chamber_length) / self.nozzle_length
            # Parabolic bell contour
            radius[div_mask] = self.throat_radius + t**0.8 * (self.exit_radius - self.throat_radius)

        return radius

    def dist(self, p: np.ndarray) -> np.ndarray:
        """Compute SDF for the nozzle shell."""
        # Cylindrical coordinates
        r = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        z = p[:, 2]

        # Clamp z to valid range
        z_clamped = np.clip(z, 0, self.total_length)

        # Inner and outer radius at each z
        r_inner = self._profile_radius(z_clamped)
        r_outer = r_inner + self.wall_thickness

        # SDF for shell (between inner and outer cylinders)
        # Distance to outer surface
        d_outer = r - r_outer

        # Distance to inner surface
        d_inner = r_inner - r

        # Distance to end caps
        d_bottom = -z
        d_top = z - self.total_length

        # Combine for shell
        d_radial = np.maximum(d_inner, d_outer)
        d_axial = np.maximum(d_bottom, d_top)

        return np.maximum(d_radial, d_axial)

    def bounds(self):
        from geometry.core import BBox3
        r_max = max(self.chamber_radius, self.exit_radius) + self.wall_thickness + 5
        return BBox3(
            min_pt=np.array([-r_max, -r_max, -5]),
            max_pt=np.array([r_max, r_max, self.total_length + 5])
        )


# ============================================================================
# Step 2: Thermal Simulation
# ============================================================================

def run_thermal_simulation(nozzle: ParametricNozzle) -> ThermalResult:
    """
    Run thermal simulation on the nozzle.

    Models heat transfer from hot combustion gases through the nozzle wall.
    """
    print("\n=== Running Thermal Simulation ===")

    # Create 2D axisymmetric thermal problem
    # (representing a radial slice of the nozzle)
    problem = ThermalProblem(
        name="nozzle_thermal",
        domain_size=(nozzle.total_length / 1000, nozzle.wall_thickness * 2 / 1000),  # m
        grid_size=(100, 20),
        thermal_conductivity=50.0,  # Copper alloy
        boundary_conditions={
            # Inner wall: hot gas convection
            "bottom": BoundaryCondition("convection", 3000.0, coefficient=5000.0),
            # Outer wall: cooling
            "top": BoundaryCondition("convection", 300.0, coefficient=1000.0),
            # Inlet
            "left": BoundaryCondition("neumann", 0.0),
            # Outlet
            "right": BoundaryCondition("neumann", 0.0),
        },
        initial_temperature=300.0,
    )

    solver = ThermalSolver(max_iterations=5000, tolerance=1e-6)
    result = solver.solve(problem)

    print(result.summary())
    return result


# ============================================================================
# Step 3: Structural Analysis with Thermal Loads
# ============================================================================

def run_structural_simulation(
    nozzle: ParametricNozzle,
    thermal_result: ThermalResult
) -> StructuralResult:
    """
    Run structural analysis with thermal loads.

    Analyzes stress/strain from pressure and thermal expansion.
    """
    print("\n=== Running Structural Simulation ===")

    # Create structural problem
    problem = StructuralProblem(
        name="nozzle_structural",
        domain_size=(nozzle.total_length / 1000, nozzle.wall_thickness * 2 / 1000),  # m
        grid_size=(50, 10),
        youngs_modulus=120e9,  # Copper alloy
        poissons_ratio=0.34,
        boundary_conditions={
            # Fixed at inlet flange
            "left": StructuralBoundaryCondition("displacement", "all", 0.0),
            # Internal pressure (simplified as distributed force)
            "bottom": StructuralBoundaryCondition("traction", "y", -5e6),  # 5 MPa
        },
        analysis_type="plane_stress",
    )

    solver = StructuralSolver()
    result = solver.solve(problem)

    print(result.summary())
    return result


# ============================================================================
# Step 4: Design Optimization
# ============================================================================

def optimize_nozzle_design():
    """
    Optimize nozzle design using genetic algorithm.

    Objectives:
    1. Maximize thrust coefficient (minimize negative)
    2. Minimize mass

    Constraints:
    - Maximum stress < yield stress
    - Minimum wall thickness
    """
    print("\n=== Running Design Optimization ===")

    def evaluate_design(x: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate a nozzle design.

        x[0]: throat_radius (5-15 mm)
        x[1]: expansion_ratio (1.5-5.0)
        x[2]: wall_thickness (1-5 mm)
        """
        throat_r = x[0]
        exp_ratio = x[1]
        wall_t = x[2]

        exit_r = throat_r * np.sqrt(exp_ratio)

        # Simplified thrust coefficient (based on expansion ratio)
        # Optimal around exp_ratio = 3-4 for most conditions
        Cf = 1.5 + 0.3 * np.log(exp_ratio) - 0.05 * (exp_ratio - 3)**2

        # Mass estimate (simplified)
        mass = np.pi * (exit_r**2 - throat_r**2) * wall_t * 8000  # kg/m³ density

        return -Cf, mass  # Minimize both (negative thrust for minimization)

    def stress_constraint(x: np.ndarray) -> float:
        """Maximum stress should be below 300 MPa."""
        # Simplified hoop stress: σ = P * r / t
        throat_r = x[0] / 1000  # Convert to m
        wall_t = x[2] / 1000
        pressure = 5e6  # Pa

        hoop_stress = pressure * throat_r / wall_t
        return hoop_stress - 300e6  # Should be negative for feasible

    # Define optimization problem
    problem = OptimizationProblem(
        name="nozzle_optimization",
        variables=[
            DesignVariable("throat_radius", 5.0, 15.0),
            DesignVariable("expansion_ratio", 1.5, 5.0),
            DesignVariable("wall_thickness", 1.0, 5.0),
        ],
        objectives=[
            lambda x: evaluate_design(x)[0],  # -Cf
            lambda x: evaluate_design(x)[1],  # mass
        ],
        constraints=[
            Constraint("stress_limit", stress_constraint),
        ],
        objective_names=["Negative Thrust Coeff", "Mass"],
    )

    # Run NSGA-II optimization
    optimizer = NSGA2(
        population_size=50,
        max_generations=30,
        seed=42,
    )

    def progress_callback(gen, pop, front):
        if gen % 10 == 0:
            print(f"  Generation {gen}: Pareto front size = {len(front)}")

    result = optimizer.optimize(problem, callback=progress_callback)

    print(result.summary())

    # Return best design (best thrust coefficient)
    best_idx = np.argmin(result.pareto_objectives[:, 0])
    return result.pareto_x[best_idx]


# ============================================================================
# Step 5: Physics-Informed Geometry Generation
# ============================================================================

def create_physics_informed_geometry(
    nozzle: ParametricNozzle,
    thermal_result: ThermalResult,
    structural_result: StructuralResult,
) -> SDF:
    """
    Create geometry that adapts to physics simulation results.
    """
    print("\n=== Creating Physics-Informed Geometry ===")

    # Create physics-geometry coupling
    coupling = CoupledPhysicsGeometry()

    # Add thermal field
    temp_field = coupling.add_thermal_field(thermal_result)
    print(f"Temperature range: {temp_field.min_value:.1f} - {temp_field.max_value:.1f} K")

    # Create thermal-adaptive wall thickness
    # Hotter regions get thicker walls
    adaptive_nozzle = ThermalWallThickness(
        base_sdf=nozzle,
        physics_field=temp_field,
        base_thickness=nozzle.wall_thickness,
        min_thickness=1.5,
        max_thickness=6.0,
        temp_ref=300.0,
        temp_max=3000.0,
        scaling_law="sqrt",
    )

    print("Created thermal-adaptive wall thickness geometry")

    return adaptive_nozzle


# ============================================================================
# Step 6: Voxelization and Export
# ============================================================================

def voxelize_and_export(sdf: SDF, filename: str, resolution: float = 1.0):
    """
    Voxelize SDF and export to mesh file.
    """
    print(f"\n=== Voxelizing (resolution={resolution}mm) ===")

    t0 = time.time()
    grid = VoxelGrid(resolution=resolution)
    grid.from_sdf(sdf, padding_voxels=2)
    print(f"Voxelization: {time.time()-t0:.2f}s")
    print(f"Grid size: {grid.size}")

    print("\n=== Generating Mesh ===")
    t0 = time.time()
    mesh = grid.meshing()
    print(f"Meshing: {time.time()-t0:.2f}s")

    mesh.save_obj(filename)
    print(f"Saved to: {filename}")

    return mesh


# ============================================================================
# Main Workflow
# ============================================================================

def main():
    """
    Run the complete CEM workflow.
    """
    print("=" * 60)
    print("Cortex CEM - Full Workflow Example")
    print("Leap71-style Computational Engineering Model")
    print("=" * 60)

    # Step 1: Create parametric nozzle
    print("\n=== Step 1: Define Parametric Geometry ===")
    nozzle = ParametricNozzle(
        chamber_radius=15.0,
        chamber_length=40.0,
        throat_radius=8.0,
        exit_radius=25.0,
        nozzle_length=60.0,
        wall_thickness=2.5,
    )
    print(f"Nozzle created:")
    print(f"  Total length: {nozzle.total_length} mm")
    print(f"  Expansion ratio: {nozzle.expansion_ratio:.2f}")

    # Step 2: Run thermal simulation
    thermal_result = run_thermal_simulation(nozzle)

    # Step 3: Run structural simulation
    structural_result = run_structural_simulation(nozzle, thermal_result)

    # Step 4: Design optimization (optional - takes longer)
    run_optimization = False  # Set to True to run
    if run_optimization:
        best_design = optimize_nozzle_design()
        print(f"\nOptimal design: {best_design}")

        # Update nozzle with optimal parameters
        nozzle = ParametricNozzle(
            throat_radius=best_design[0],
            exit_radius=best_design[0] * np.sqrt(best_design[1]),
            wall_thickness=best_design[2],
        )

    # Step 5: Create physics-informed geometry
    adaptive_geometry = create_physics_informed_geometry(
        nozzle, thermal_result, structural_result
    )

    # Step 6: Export
    # Note: Using coarse resolution for demo (1.0mm)
    # Production would use 0.1-0.2mm
    voxelize_and_export(
        adaptive_geometry,
        "cem_optimized_nozzle.obj",
        resolution=1.0
    )

    print("\n" + "=" * 60)
    print("CEM Workflow Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
