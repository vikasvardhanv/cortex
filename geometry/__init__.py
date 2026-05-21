"""
Cortex Geometry Engine

Provides SDF-based geometry generation, CSG operations, voxelization,
and mesh export. This is the foundation for generating manufacturable
3D geometry from computational engineering designs.

Based on the original MyCEM geometry kernel, inspired by Leap71's PicoGK.

Integration module provides physics-informed geometry modification:
- ThermalWallThickness: Vary wall thickness based on temperature
- StressOptimizedTopology: Remove low-stress material
- FlowGuidedSurface: Shape surfaces based on flow fields
- AdaptiveLattice: Physics-driven lattice structures
"""

from geometry.core import (
    SDF,
    Union,
    Difference,
    Intersection,
    Offset,
    SmoothUnion,
    BBox3,
    Mesh,
    Translate,
    RotateMC,
    VoxelGrid,
)

from geometry.shapes import (
    Sphere,
    Box,
    Cylinder,
    Torus,
    Gyroid,
    Revolve,
)

from geometry.integration import (
    PhysicsField,
    PhysicsInformedSDF,
    ThermalWallThickness,
    StressOptimizedTopology,
    FlowGuidedSurface,
    AdaptiveLattice,
    CoupledPhysicsGeometry,
)

__all__ = [
    # Core
    "SDF",
    "Union",
    "Difference",
    "Intersection",
    "Offset",
    "SmoothUnion",
    "BBox3",
    "Mesh",
    "Translate",
    "RotateMC",
    "VoxelGrid",
    # Shapes
    "Sphere",
    "Box",
    "Cylinder",
    "Torus",
    "Gyroid",
    "Revolve",
    # Physics Integration
    "PhysicsField",
    "PhysicsInformedSDF",
    "ThermalWallThickness",
    "StressOptimizedTopology",
    "FlowGuidedSurface",
    "AdaptiveLattice",
    "CoupledPhysicsGeometry",
]
