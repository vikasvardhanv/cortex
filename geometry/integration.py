"""
Geometry Integration for Cortex CEM

Connects physics solver results to SDF geometry generation.
Enables physics-informed geometry modification:
- Thermal-driven wall thickness variation
- Stress-driven topology optimization
- Flow-driven surface shaping

This is the core of Leap71-style Computational Engineering Models
where physics simulations drive geometric design decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import numpy as np
from scipy.interpolate import RegularGridInterpolator, RBFInterpolator
from scipy.ndimage import gaussian_filter

from geometry.core import SDF, BBox3, Union as SDFUnion, Difference, Offset


@dataclass
class PhysicsField:
    """
    A scalar or vector field from physics simulation.

    Provides interpolation for querying values at arbitrary points.
    """
    name: str
    values: np.ndarray  # [nx, ny] or [nx, ny, nz] or [nx, ny, ndim]
    x_coords: np.ndarray
    y_coords: np.ndarray
    z_coords: Optional[np.ndarray] = None
    field_type: str = "scalar"  # "scalar" or "vector"

    _interpolator: Optional[RegularGridInterpolator] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize interpolator."""
        if self.z_coords is not None:
            points = (self.x_coords, self.y_coords, self.z_coords)
        else:
            points = (self.x_coords, self.y_coords)

        self._interpolator = RegularGridInterpolator(
            points, self.values,
            method='linear',
            bounds_error=False,
            fill_value=None  # Extrapolate
        )

    def query(self, points: np.ndarray) -> np.ndarray:
        """
        Query field values at arbitrary points.

        Args:
            points: [N, 2] or [N, 3] array of query points

        Returns:
            [N,] or [N, ndim] array of field values
        """
        return self._interpolator(points)

    @property
    def min_value(self) -> float:
        return float(np.min(self.values))

    @property
    def max_value(self) -> float:
        return float(np.max(self.values))

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bounding box of the field."""
        if self.z_coords is not None:
            min_pt = np.array([self.x_coords[0], self.y_coords[0], self.z_coords[0]])
            max_pt = np.array([self.x_coords[-1], self.y_coords[-1], self.z_coords[-1]])
        else:
            min_pt = np.array([self.x_coords[0], self.y_coords[0], 0])
            max_pt = np.array([self.x_coords[-1], self.y_coords[-1], 0])
        return min_pt, max_pt


class PhysicsInformedSDF(SDF):
    """
    SDF that modifies geometry based on physics field values.

    Base class for physics-driven geometry modification.
    """

    def __init__(self, base_sdf: SDF, physics_field: PhysicsField):
        self.base_sdf = base_sdf
        self.physics_field = physics_field

    def dist(self, p: np.ndarray) -> np.ndarray:
        """Default: return base SDF distance."""
        return self.base_sdf.dist(p)

    def bounds(self) -> BBox3:
        return self.base_sdf.bounds()


class ThermalWallThickness(PhysicsInformedSDF):
    """
    Modifies wall thickness based on temperature field.

    Hotter regions get thicker walls for thermal protection.
    Cooler regions can have thinner walls to save mass.

    Usage:
        thermal_result = thermal_solver.solve(problem)
        temp_field = PhysicsField(
            name="temperature",
            values=thermal_result.temperature,
            x_coords=thermal_result.x_coords,
            y_coords=thermal_result.y_coords,
        )
        modified_sdf = ThermalWallThickness(
            base_sdf=nozzle_outer,
            physics_field=temp_field,
            base_thickness=2.0,
            min_thickness=1.0,
            max_thickness=5.0,
            temp_ref=300.0,
            temp_max=2000.0,
        )
    """

    def __init__(
        self,
        base_sdf: SDF,
        physics_field: PhysicsField,
        base_thickness: float = 2.0,
        min_thickness: float = 0.5,
        max_thickness: float = 10.0,
        temp_ref: float = 300.0,  # Reference temperature (K)
        temp_max: float = 2000.0,  # Max temperature for scaling
        scaling_law: str = "linear",  # "linear", "sqrt", "log"
    ):
        super().__init__(base_sdf, physics_field)
        self.base_thickness = base_thickness
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.temp_ref = temp_ref
        self.temp_max = temp_max
        self.scaling_law = scaling_law

    def _compute_thickness(self, temperature: np.ndarray) -> np.ndarray:
        """Compute wall thickness based on local temperature."""
        # Normalize temperature
        t_norm = (temperature - self.temp_ref) / (self.temp_max - self.temp_ref)
        t_norm = np.clip(t_norm, 0, 1)

        # Apply scaling law
        if self.scaling_law == "linear":
            scale = t_norm
        elif self.scaling_law == "sqrt":
            scale = np.sqrt(t_norm)
        elif self.scaling_law == "log":
            scale = np.log1p(t_norm * (np.e - 1))
        else:
            scale = t_norm

        # Interpolate thickness
        thickness = self.base_thickness + scale * (self.max_thickness - self.base_thickness)
        return np.clip(thickness, self.min_thickness, self.max_thickness)

    def dist(self, p: np.ndarray) -> np.ndarray:
        """
        Compute SDF with thermal-dependent wall thickness.

        The wall thickness varies based on local temperature.
        """
        # Get base distance
        base_dist = self.base_sdf.dist(p)

        # Query temperature at points (use 2D projection for 3D points)
        if p.shape[1] == 3:
            query_pts = p[:, :2]  # Project to 2D for 2D fields
        else:
            query_pts = p

        temperature = self.physics_field.query(query_pts)

        # Handle NaN from extrapolation
        temperature = np.nan_to_num(temperature, nan=self.temp_ref)

        # Compute local thickness
        thickness = self._compute_thickness(temperature)

        # Modify SDF: offset inward by thickness variation
        thickness_offset = thickness - self.base_thickness

        # For points on the outer surface, shift them based on thickness
        # This effectively creates a shell with varying thickness
        return base_dist - thickness_offset


class StressOptimizedTopology(PhysicsInformedSDF):
    """
    Modifies geometry based on stress field for topology optimization.

    Low-stress regions can be removed to reduce mass.
    High-stress regions need reinforcement.

    Implements a SIMP-like (Solid Isotropic Material with Penalization)
    approach for topology optimization.
    """

    def __init__(
        self,
        base_sdf: SDF,
        physics_field: PhysicsField,  # von Mises stress field
        stress_threshold: float = 0.3,  # Fraction of max stress below which to remove
        safety_factor: float = 1.5,
        min_density: float = 0.01,
        penalization: float = 3.0,  # SIMP penalization exponent
        smoothing_radius: float = 0.0,
    ):
        super().__init__(base_sdf, physics_field)
        self.stress_threshold = stress_threshold
        self.safety_factor = safety_factor
        self.min_density = min_density
        self.penalization = penalization
        self.smoothing_radius = smoothing_radius

        # Normalize stress field
        self._max_stress = physics_field.max_value
        self._smoothed_values = None

        if smoothing_radius > 0:
            # Apply Gaussian smoothing for minimum member size control
            sigma = smoothing_radius / (physics_field.x_coords[1] - physics_field.x_coords[0])
            self._smoothed_values = gaussian_filter(physics_field.values, sigma=sigma)

    def _compute_density(self, stress: np.ndarray) -> np.ndarray:
        """
        Compute material density based on stress.

        Returns values between 0 (void) and 1 (solid).
        """
        # Normalize stress
        stress_norm = stress / (self._max_stress * self.safety_factor)

        # Apply threshold: regions below threshold are candidates for removal
        density = np.where(
            stress_norm < self.stress_threshold,
            self.min_density,
            np.clip(stress_norm, self.min_density, 1.0)
        )

        # Apply SIMP penalization
        density = density ** (1.0 / self.penalization)

        return density

    def dist(self, p: np.ndarray) -> np.ndarray:
        """
        Compute SDF with stress-based topology modification.

        Low-stress regions are effectively removed (large positive distance).
        """
        base_dist = self.base_sdf.dist(p)

        # Query stress field
        if p.shape[1] == 3:
            query_pts = p[:, :2]
        else:
            query_pts = p

        stress = self.physics_field.query(query_pts)
        stress = np.nan_to_num(stress, nan=0.0)

        # Compute density
        density = self._compute_density(stress)

        # Modify SDF: void regions get large positive distance
        # Solid regions (density=1) keep original SDF
        # Intermediate values create gradient
        void_offset = (1.0 - density) * 10.0  # Large offset for voids

        # Only apply to points that are inside the base geometry
        inside_mask = base_dist < 0
        modified_dist = np.where(
            inside_mask,
            base_dist + void_offset,
            base_dist
        )

        return modified_dist


class FlowGuidedSurface(PhysicsInformedSDF):
    """
    Shapes surface based on flow velocity/pressure fields.

    Used for aerodynamic/hydrodynamic optimization:
    - Streamline channels based on flow direction
    - Optimize cross-sections for pressure drop
    - Create flow-following lattice structures
    """

    def __init__(
        self,
        base_sdf: SDF,
        velocity_field: PhysicsField,  # Vector field [nx, ny, 2]
        target_velocity: float = 1.0,
        expansion_rate: float = 0.5,  # How much to expand in slow regions
        contraction_rate: float = 0.3,  # How much to contract in fast regions
    ):
        super().__init__(base_sdf, velocity_field)
        self.target_velocity = target_velocity
        self.expansion_rate = expansion_rate
        self.contraction_rate = contraction_rate

    def dist(self, p: np.ndarray) -> np.ndarray:
        """Modify surface based on local flow velocity."""
        base_dist = self.base_sdf.dist(p)

        if p.shape[1] == 3:
            query_pts = p[:, :2]
        else:
            query_pts = p

        # Query velocity magnitude
        velocity = self.physics_field.query(query_pts)
        if velocity.ndim == 2:
            velocity_mag = np.linalg.norm(velocity, axis=-1)
        else:
            velocity_mag = velocity

        velocity_mag = np.nan_to_num(velocity_mag, nan=self.target_velocity)

        # Compute surface modification
        velocity_ratio = velocity_mag / self.target_velocity

        # Slow regions: expand to increase flow area
        # Fast regions: contract to slow flow
        offset = np.where(
            velocity_ratio < 1.0,
            self.expansion_rate * (1.0 - velocity_ratio),
            -self.contraction_rate * (velocity_ratio - 1.0)
        )

        return base_dist - offset


class AdaptiveLattice(SDF):
    """
    Creates a lattice structure with physics-adaptive properties.

    Cell size and strut thickness vary based on physics fields.
    """

    def __init__(
        self,
        bounding_sdf: SDF,  # Outer boundary
        physics_field: PhysicsField,
        base_cell_size: float = 5.0,
        min_cell_size: float = 2.0,
        max_cell_size: float = 10.0,
        base_strut_radius: float = 0.5,
        min_strut_radius: float = 0.2,
        max_strut_radius: float = 1.5,
        lattice_type: str = "octet",  # "octet", "gyroid", "diamond"
    ):
        self.bounding_sdf = bounding_sdf
        self.physics_field = physics_field
        self.base_cell_size = base_cell_size
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size
        self.base_strut_radius = base_strut_radius
        self.min_strut_radius = min_strut_radius
        self.max_strut_radius = max_strut_radius
        self.lattice_type = lattice_type

    def _gyroid_sdf(self, p: np.ndarray, cell_size: np.ndarray) -> np.ndarray:
        """Gyroid TPMS surface."""
        # Normalized coordinates
        x = p[:, 0] * 2 * np.pi / cell_size
        y = p[:, 1] * 2 * np.pi / cell_size

        if p.shape[1] == 3:
            z = p[:, 2] * 2 * np.pi / cell_size
            return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
        else:
            return np.sin(x) * np.cos(y)

    def _diamond_sdf(self, p: np.ndarray, cell_size: np.ndarray) -> np.ndarray:
        """Diamond TPMS surface."""
        x = p[:, 0] * 2 * np.pi / cell_size
        y = p[:, 1] * 2 * np.pi / cell_size

        if p.shape[1] == 3:
            z = p[:, 2] * 2 * np.pi / cell_size
            return (np.sin(x) * np.sin(y) * np.sin(z) +
                    np.sin(x) * np.cos(y) * np.cos(z) +
                    np.cos(x) * np.sin(y) * np.cos(z) +
                    np.cos(x) * np.cos(y) * np.sin(z))
        else:
            return np.sin(x) * np.sin(y)

    def dist(self, p: np.ndarray) -> np.ndarray:
        """Compute SDF for adaptive lattice."""
        # Get bounding SDF
        bound_dist = self.bounding_sdf.dist(p)

        # Query physics field to determine local properties
        if p.shape[1] == 3:
            query_pts = p[:, :2]
        else:
            query_pts = p

        field_value = self.physics_field.query(query_pts)
        field_value = np.nan_to_num(field_value, nan=0.5)

        # Normalize field value
        field_norm = (field_value - self.physics_field.min_value) / (
            self.physics_field.max_value - self.physics_field.min_value + 1e-10
        )

        # Compute local cell size (high field -> small cells for better resolution)
        cell_size = self.max_cell_size - field_norm * (self.max_cell_size - self.min_cell_size)

        # Compute local strut radius (high field -> thick struts for strength)
        strut_radius = self.min_strut_radius + field_norm * (self.max_strut_radius - self.min_strut_radius)

        # Compute lattice SDF
        if self.lattice_type == "gyroid":
            lattice_dist = self._gyroid_sdf(p, cell_size) - strut_radius
        elif self.lattice_type == "diamond":
            lattice_dist = self._diamond_sdf(p, cell_size) - strut_radius
        else:  # octet
            lattice_dist = self._gyroid_sdf(p, cell_size) - strut_radius

        # Intersect with bounding volume
        return np.maximum(bound_dist, -lattice_dist)

    def bounds(self) -> BBox3:
        return self.bounding_sdf.bounds()


class CoupledPhysicsGeometry:
    """
    Manages the coupling between multiple physics solvers and geometry.

    Orchestrates the iterative design process:
    1. Solve physics on current geometry
    2. Extract relevant fields
    3. Modify geometry based on physics
    4. Repeat until convergence
    """

    def __init__(self):
        self.physics_fields: Dict[str, PhysicsField] = {}
        self.history: List[Dict] = []

    def add_field(self, name: str, field: PhysicsField):
        """Add a physics field from solver results."""
        self.physics_fields[name] = field

    def add_thermal_field(self, thermal_result, name: str = "temperature"):
        """Add thermal field from ThermalResult."""
        field = PhysicsField(
            name=name,
            values=thermal_result.temperature,
            x_coords=thermal_result.x_coords,
            y_coords=thermal_result.y_coords,
        )
        self.physics_fields[name] = field
        return field

    def add_structural_field(self, structural_result, field_type: str = "von_mises"):
        """Add structural field from StructuralResult."""
        if field_type == "von_mises":
            values = structural_result.stress_vm
        elif field_type == "displacement":
            values = np.linalg.norm(structural_result.displacement, axis=1)
        else:
            values = structural_result.get_field(field_type)

        # Reshape to grid if needed
        if values.ndim == 1:
            # Assume it's element-based, need to map to node grid
            # For simplicity, we'll interpolate
            from scipy.interpolate import griddata
            nx = len(np.unique(structural_result.x_coords))
            ny = len(np.unique(structural_result.y_coords))

            x_grid = np.linspace(structural_result.x_coords.min(),
                                structural_result.x_coords.max(), nx)
            y_grid = np.linspace(structural_result.y_coords.min(),
                                structural_result.y_coords.max(), ny)

            xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')

            # For element-centered values, use centroid coordinates
            points = np.column_stack([structural_result.x_coords,
                                      structural_result.y_coords])
            values = griddata(points, values, (xx, yy), method='nearest')

        field = PhysicsField(
            name=field_type,
            values=values,
            x_coords=structural_result.x_coords if values.ndim == 1 else x_grid,
            y_coords=structural_result.y_coords if values.ndim == 1 else y_grid,
        )
        self.physics_fields[field_type] = field
        return field

    def add_fluid_field(self, fluid_result, field_type: str = "velocity"):
        """Add fluid field from FluidResult."""
        if field_type == "velocity":
            values = fluid_result.velocity_magnitude
        elif field_type == "pressure":
            values = fluid_result.pressure
        else:
            values = fluid_result.get_field(field_type)

        field = PhysicsField(
            name=field_type,
            values=values,
            x_coords=fluid_result.x_coords,
            y_coords=fluid_result.y_coords,
        )
        self.physics_fields[field_type] = field
        return field

    def create_thermal_wall(
        self,
        base_sdf: SDF,
        temperature_field: str = "temperature",
        **kwargs
    ) -> ThermalWallThickness:
        """Create thermal-adaptive wall from temperature field."""
        if temperature_field not in self.physics_fields:
            raise ValueError(f"Field '{temperature_field}' not found")

        return ThermalWallThickness(
            base_sdf=base_sdf,
            physics_field=self.physics_fields[temperature_field],
            **kwargs
        )

    def create_stress_topology(
        self,
        base_sdf: SDF,
        stress_field: str = "von_mises",
        **kwargs
    ) -> StressOptimizedTopology:
        """Create stress-optimized topology from stress field."""
        if stress_field not in self.physics_fields:
            raise ValueError(f"Field '{stress_field}' not found")

        return StressOptimizedTopology(
            base_sdf=base_sdf,
            physics_field=self.physics_fields[stress_field],
            **kwargs
        )

    def create_flow_surface(
        self,
        base_sdf: SDF,
        velocity_field: str = "velocity",
        **kwargs
    ) -> FlowGuidedSurface:
        """Create flow-guided surface from velocity field."""
        if velocity_field not in self.physics_fields:
            raise ValueError(f"Field '{velocity_field}' not found")

        return FlowGuidedSurface(
            base_sdf=base_sdf,
            velocity_field=self.physics_fields[velocity_field],
            **kwargs
        )

    def create_adaptive_lattice(
        self,
        bounding_sdf: SDF,
        driving_field: str = "temperature",
        **kwargs
    ) -> AdaptiveLattice:
        """Create physics-adaptive lattice structure."""
        if driving_field not in self.physics_fields:
            raise ValueError(f"Field '{driving_field}' not found")

        return AdaptiveLattice(
            bounding_sdf=bounding_sdf,
            physics_field=self.physics_fields[driving_field],
            **kwargs
        )


def thermal_to_geometry_example():
    """
    Example: Create a rocket nozzle with thermal-adaptive wall thickness.

    This demonstrates the Leap71 CEM workflow:
    1. Define base geometry
    2. Run thermal simulation
    3. Modify geometry based on thermal results
    4. Export for manufacturing
    """
    # This is a conceptual example showing the workflow
    example_code = '''
    from geometry import Cylinder, Cone, SmoothUnion, Difference, Offset
    from solvers.thermal import ThermalSolver, ThermalProblem
    from geometry.integration import (
        CoupledPhysicsGeometry, ThermalWallThickness, PhysicsField
    )

    # 1. Define base nozzle geometry
    chamber = Cylinder(radius=15.0, height=40.0, center=[0, 0, 20])
    bell = Cone(angle=np.radians(15), height=60.0, center=[0, 0, 40])
    nozzle_outer = SmoothUnion(chamber, bell, k=5.0)

    # 2. Run thermal simulation
    thermal_problem = ThermalProblem(
        name="nozzle_thermal",
        domain_size=(0.1, 0.1),
        grid_size=(100, 100),
        thermal_conductivity=50.0,
        boundary_conditions={
            "left": BoundaryCondition("dirichlet", 2500.0),   # Hot gas
            "right": BoundaryCondition("convection", 300.0, 100.0),  # Ambient
        }
    )

    thermal_solver = ThermalSolver()
    thermal_result = thermal_solver.solve(thermal_problem)

    # 3. Create physics-geometry coupling
    coupling = CoupledPhysicsGeometry()
    temp_field = coupling.add_thermal_field(thermal_result)

    # 4. Create adaptive geometry
    adaptive_nozzle = coupling.create_thermal_wall(
        base_sdf=nozzle_outer,
        temperature_field="temperature",
        base_thickness=2.0,
        min_thickness=1.0,
        max_thickness=8.0,
        temp_ref=300.0,
        temp_max=2500.0,
    )

    # 5. Voxelize and export
    from geometry.core import VoxelGrid
    grid = VoxelGrid(resolution=0.5)
    grid.from_sdf(adaptive_nozzle)
    mesh = grid.meshing()
    mesh.save_obj("thermal_adaptive_nozzle.obj")
    '''
    return example_code
