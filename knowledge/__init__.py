"""
Knowledge Base for Cortex CEM

Contains:
- MaterialsDB: Database of engineering materials and their properties
- PhysicsRules: Physical laws and equations for engineering calculations
- DesignPatterns: Common engineering design patterns and constraints
"""

from .materials import MaterialsDB, Material
from .physics_rules import PhysicsRules
from .design_patterns import DesignPatterns

__all__ = [
    "MaterialsDB",
    "Material",
    "PhysicsRules",
    "DesignPatterns",
]
