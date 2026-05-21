"""
Materials Database for Cortex CEM

Contains properties of common engineering materials used in:
- Thermal analysis (conductivity, specific heat, melting point)
- Structural analysis (Young's modulus, yield strength, density)
- Aerospace/High-temp applications (thermal expansion, max service temp)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import json


@dataclass
class Material:
    """
    Engineering material with physical properties.

    All units are SI:
    - Temperature: Kelvin (K)
    - Thermal conductivity: W/(m·K)
    - Specific heat: J/(kg·K)
    - Density: kg/m³
    - Young's modulus: Pa (Pascals)
    - Yield strength: Pa
    - Thermal expansion: 1/K
    """
    name: str
    category: str  # metal, ceramic, composite, polymer

    # Thermal properties
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    melting_point: float  # K
    max_service_temp: float  # K (maximum usable temperature)
    thermal_expansion: float  # 1/K (coefficient of linear thermal expansion)

    # Mechanical properties
    density: float  # kg/m³
    youngs_modulus: float  # Pa
    yield_strength: float  # Pa
    poissons_ratio: float  # dimensionless

    # Additional properties
    description: str = ""
    applications: List[str] = field(default_factory=list)

    def thermal_diffusivity(self) -> float:
        """Calculate thermal diffusivity: α = k / (ρ * cp)"""
        return self.thermal_conductivity / (self.density * self.specific_heat)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "category": self.category,
            "thermal_conductivity": self.thermal_conductivity,
            "specific_heat": self.specific_heat,
            "melting_point": self.melting_point,
            "max_service_temp": self.max_service_temp,
            "thermal_expansion": self.thermal_expansion,
            "density": self.density,
            "youngs_modulus": self.youngs_modulus,
            "yield_strength": self.yield_strength,
            "poissons_ratio": self.poissons_ratio,
            "description": self.description,
            "applications": self.applications,
        }


class MaterialsDB:
    """
    Database of engineering materials.

    Usage:
        db = MaterialsDB()
        steel = db.get("stainless_steel_316")
        print(steel.thermal_conductivity)

        # Find materials for high-temp applications
        candidates = db.find_by_max_temp(min_temp=1500)
    """

    def __init__(self):
        self._materials: Dict[str, Material] = {}
        self._load_default_materials()

    def _load_default_materials(self):
        """Load built-in materials database."""

        # === METALS ===

        self.add(Material(
            name="aluminum_6061",
            category="metal",
            thermal_conductivity=167.0,
            specific_heat=896.0,
            melting_point=855.0,
            max_service_temp=573.0,
            thermal_expansion=23.6e-6,
            density=2700.0,
            youngs_modulus=68.9e9,
            yield_strength=276e6,
            poissons_ratio=0.33,
            description="Common aerospace aluminum alloy",
            applications=["aircraft structures", "automotive", "general engineering"]
        ))

        self.add(Material(
            name="stainless_steel_316",
            category="metal",
            thermal_conductivity=16.3,
            specific_heat=500.0,
            melting_point=1673.0,
            max_service_temp=1143.0,
            thermal_expansion=16.0e-6,
            density=8000.0,
            youngs_modulus=193e9,
            yield_strength=290e6,
            poissons_ratio=0.27,
            description="Corrosion-resistant stainless steel",
            applications=["chemical processing", "marine", "medical"]
        ))

        self.add(Material(
            name="inconel_718",
            category="metal",
            thermal_conductivity=11.4,
            specific_heat=435.0,
            melting_point=1609.0,
            max_service_temp=973.0,
            thermal_expansion=13.0e-6,
            density=8190.0,
            youngs_modulus=200e9,
            yield_strength=1100e6,
            poissons_ratio=0.29,
            description="Nickel-based superalloy for high-temp applications",
            applications=["jet engines", "rocket engines", "gas turbines"]
        ))

        self.add(Material(
            name="titanium_ti6al4v",
            category="metal",
            thermal_conductivity=6.7,
            specific_heat=526.0,
            melting_point=1933.0,
            max_service_temp=673.0,
            thermal_expansion=8.6e-6,
            density=4430.0,
            youngs_modulus=113.8e9,
            yield_strength=880e6,
            poissons_ratio=0.34,
            description="Aerospace titanium alloy",
            applications=["aerospace", "medical implants", "marine"]
        ))

        self.add(Material(
            name="copper_c11000",
            category="metal",
            thermal_conductivity=391.0,
            specific_heat=385.0,
            melting_point=1356.0,
            max_service_temp=473.0,
            thermal_expansion=16.5e-6,
            density=8940.0,
            youngs_modulus=117e9,
            yield_strength=69e6,
            poissons_ratio=0.34,
            description="Pure copper, excellent thermal/electrical conductor",
            applications=["heat exchangers", "electrical", "rocket combustion chambers"]
        ))

        # === CERAMICS ===

        self.add(Material(
            name="silicon_carbide",
            category="ceramic",
            thermal_conductivity=120.0,
            specific_heat=750.0,
            melting_point=3003.0,
            max_service_temp=1873.0,
            thermal_expansion=4.0e-6,
            density=3210.0,
            youngs_modulus=410e9,
            yield_strength=3440e6,  # Compressive strength
            poissons_ratio=0.14,
            description="High-temp ceramic for extreme environments",
            applications=["heat shields", "armor", "semiconductor"]
        ))

        self.add(Material(
            name="alumina_al2o3",
            category="ceramic",
            thermal_conductivity=35.0,
            specific_heat=880.0,
            melting_point=2345.0,
            max_service_temp=1973.0,
            thermal_expansion=8.1e-6,
            density=3950.0,
            youngs_modulus=380e9,
            yield_strength=2600e6,
            poissons_ratio=0.22,
            description="Aluminum oxide ceramic",
            applications=["thermal barriers", "electrical insulators", "wear parts"]
        ))

        # === COMPOSITES ===

        self.add(Material(
            name="carbon_carbon",
            category="composite",
            thermal_conductivity=50.0,  # Varies with direction
            specific_heat=710.0,
            melting_point=3823.0,  # Sublimation point
            max_service_temp=2273.0,  # In inert atmosphere
            thermal_expansion=1.0e-6,
            density=1800.0,
            youngs_modulus=70e9,
            yield_strength=300e6,
            poissons_ratio=0.25,
            description="Carbon fiber reinforced carbon matrix",
            applications=["heat shields", "rocket nozzles", "brake discs"]
        ))

        self.add(Material(
            name="reinforced_carbon_carbon",
            category="composite",
            thermal_conductivity=40.0,
            specific_heat=1260.0,
            melting_point=3773.0,
            max_service_temp=1923.0,
            thermal_expansion=0.5e-6,
            density=1650.0,
            youngs_modulus=100e9,
            yield_strength=350e6,
            poissons_ratio=0.3,
            description="RCC - Space Shuttle heat shield material",
            applications=["re-entry heat shields", "leading edges"]
        ))

        # === ABLATIVES ===

        self.add(Material(
            name="pica",
            category="composite",
            thermal_conductivity=0.21,
            specific_heat=1674.0,
            melting_point=3773.0,
            max_service_temp=2773.0,
            thermal_expansion=1.5e-6,
            density=270.0,
            youngs_modulus=0.1e9,
            yield_strength=5e6,
            poissons_ratio=0.3,
            description="Phenolic Impregnated Carbon Ablator - Mars missions",
            applications=["planetary entry heat shields", "ablative protection"]
        ))

    def add(self, material: Material):
        """Add a material to the database."""
        key = material.name.lower().replace(" ", "_")
        self._materials[key] = material

    def get(self, name: str) -> Optional[Material]:
        """Get a material by name."""
        key = name.lower().replace(" ", "_")
        return self._materials.get(key)

    def list_all(self) -> List[str]:
        """List all material names."""
        return list(self._materials.keys())

    def list_by_category(self, category: str) -> List[Material]:
        """List materials by category (metal, ceramic, composite)."""
        return [m for m in self._materials.values() if m.category == category]

    def find_by_max_temp(self, min_temp: float) -> List[Material]:
        """Find materials that can operate above a given temperature (K)."""
        return [m for m in self._materials.values() if m.max_service_temp >= min_temp]

    def find_by_conductivity(self, min_k: float = 0, max_k: float = float('inf')) -> List[Material]:
        """Find materials within a thermal conductivity range."""
        return [m for m in self._materials.values()
                if min_k <= m.thermal_conductivity <= max_k]

    def recommend_for_application(self, application: str) -> List[Material]:
        """Find materials suitable for a given application."""
        app_lower = application.lower()
        return [m for m in self._materials.values()
                if any(app_lower in a.lower() for a in m.applications)]

    def to_json(self) -> str:
        """Export database as JSON."""
        return json.dumps({k: v.to_dict() for k, v in self._materials.items()}, indent=2)

    def summary(self) -> str:
        """Print a summary of available materials."""
        lines = ["=== Cortex Materials Database ===", ""]
        for category in ["metal", "ceramic", "composite"]:
            materials = self.list_by_category(category)
            if materials:
                lines.append(f"{category.upper()}S ({len(materials)}):")
                for m in materials:
                    lines.append(f"  - {m.name}: k={m.thermal_conductivity} W/(m·K), Tmax={m.max_service_temp}K")
                lines.append("")
        return "\n".join(lines)
