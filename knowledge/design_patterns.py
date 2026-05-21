"""
Design Patterns for Cortex CEM

Contains common engineering design patterns, rules of thumb,
and constraints used in computational engineering.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class DesignDomain(Enum):
    """Engineering design domains."""
    AEROSPACE = "aerospace"
    THERMAL_MANAGEMENT = "thermal_management"
    STRUCTURAL = "structural"
    PROPULSION = "propulsion"
    GENERAL = "general"


@dataclass
class DesignPattern:
    """A reusable engineering design pattern."""
    name: str
    domain: DesignDomain
    description: str
    rules: List[str]
    parameters: Dict[str, str]
    example_applications: List[str]


class DesignPatterns:
    """
    Collection of engineering design patterns and rules of thumb.

    These encode domain knowledge that helps guide the design process.
    """

    def __init__(self):
        self._patterns: Dict[str, DesignPattern] = {}
        self._load_default_patterns()

    def _load_default_patterns(self):
        """Load default design patterns."""

        self.add(DesignPattern(
            name="heat_shield_ablative",
            domain=DesignDomain.AEROSPACE,
            description="Ablative heat shield for atmospheric re-entry",
            rules=[
                "Use ablative material (PICA, carbon-carbon) for extreme heat flux",
                "Thickness determined by heat load integral over re-entry trajectory",
                "Surface recession rate must not expose structure",
                "Consider charring depth and virgin material properties",
                "Typical heat flux at re-entry: 100-1000 W/cm² (depends on velocity)",
            ],
            parameters={
                "heat_flux_peak": "Maximum heat flux during re-entry (W/m²)",
                "entry_velocity": "Atmospheric entry velocity (m/s)",
                "entry_angle": "Entry angle (degrees from horizontal)",
                "duration": "Total heating duration (s)",
            },
            example_applications=["Mars entry capsule", "Space Shuttle leading edge", "Crew capsule"]
        ))

        self.add(DesignPattern(
            name="heat_sink_plate",
            domain=DesignDomain.THERMAL_MANAGEMENT,
            description="Simple heat sink for thermal management",
            rules=[
                "Use high thermal conductivity material (Al, Cu)",
                "Maximize surface area with fins if convective cooling",
                "Thermal resistance: R = L/(k*A) for conduction",
                "Check thermal mass for transient applications",
                "Ensure good thermal contact at interfaces",
            ],
            parameters={
                "heat_load": "Heat to be dissipated (W)",
                "max_temperature": "Maximum allowable temperature (K)",
                "ambient_temperature": "Surrounding temperature (K)",
                "available_volume": "Space available for heat sink",
            },
            example_applications=["Electronics cooling", "LED thermal management", "Power electronics"]
        ))

        self.add(DesignPattern(
            name="rocket_nozzle",
            domain=DesignDomain.PROPULSION,
            description="Convergent-divergent nozzle for rocket propulsion",
            rules=[
                "Use De Laval nozzle geometry for supersonic expansion",
                "Throat area determines mass flow rate",
                "Exit area ratio determines expansion (and thrust)",
                "Wall cooling required: regenerative, film, or ablative",
                "Consider thermal stress at throat (highest heat flux)",
            ],
            parameters={
                "chamber_pressure": "Combustion chamber pressure (Pa)",
                "chamber_temperature": "Combustion temperature (K)",
                "mass_flow_rate": "Propellant mass flow (kg/s)",
                "expansion_ratio": "Exit area / throat area",
            },
            example_applications=["Liquid rocket engine", "Solid rocket motor", "Electric thruster"]
        ))

        self.add(DesignPattern(
            name="structural_bracket",
            domain=DesignDomain.STRUCTURAL,
            description="Load-bearing structural bracket",
            rules=[
                "Safety factor typically 1.5-2.0 for aerospace, 2.5+ for general",
                "Check both yield and fatigue if cyclic loading",
                "Consider stress concentrations at corners (add fillets)",
                "Buckling check for thin-walled sections",
                "Weight optimization: topology optimization beneficial",
            ],
            parameters={
                "applied_load": "Maximum applied force (N)",
                "load_type": "Static, cyclic, or impact",
                "mounting_points": "Number and location of attachments",
                "weight_target": "Maximum allowable mass (kg)",
            },
            example_applications=["Engine mount", "Avionics bracket", "Solar panel mount"]
        ))

        self.add(DesignPattern(
            name="heat_exchanger_plate",
            domain=DesignDomain.THERMAL_MANAGEMENT,
            description="Plate-type heat exchanger",
            rules=[
                "Counter-flow arrangement most efficient",
                "Effectiveness depends on NTU and capacity ratio",
                "Consider pressure drop vs heat transfer tradeoff",
                "Fouling factor reduces performance over time",
                "LMTD method for sizing: Q = U*A*LMTD",
            ],
            parameters={
                "hot_fluid_inlet_temp": "Hot side inlet temperature (K)",
                "cold_fluid_inlet_temp": "Cold side inlet temperature (K)",
                "heat_duty": "Required heat transfer (W)",
                "max_pressure_drop": "Allowable pressure drop (Pa)",
            },
            example_applications=["Rocket engine cooling", "HVAC", "Industrial process"]
        ))

    def add(self, pattern: DesignPattern):
        """Add a design pattern."""
        self._patterns[pattern.name] = pattern

    def get(self, name: str) -> Optional[DesignPattern]:
        """Get a pattern by name."""
        return self._patterns.get(name)

    def list_all(self) -> List[str]:
        """List all pattern names."""
        return list(self._patterns.keys())

    def by_domain(self, domain: DesignDomain) -> List[DesignPattern]:
        """Get patterns for a specific domain."""
        return [p for p in self._patterns.values() if p.domain == domain]

    def find_relevant(self, keywords: List[str]) -> List[DesignPattern]:
        """Find patterns relevant to given keywords."""
        keywords_lower = [k.lower() for k in keywords]
        results = []

        for pattern in self._patterns.values():
            # Check if any keyword matches pattern name, description, or applications
            text = f"{pattern.name} {pattern.description} {' '.join(pattern.example_applications)}".lower()
            if any(kw in text for kw in keywords_lower):
                results.append(pattern)

        return results

    def get_rules_for_problem(self, problem_description: str) -> List[str]:
        """Extract relevant rules for a given problem description."""
        keywords = problem_description.lower().split()
        relevant_patterns = self.find_relevant(keywords)

        all_rules = []
        for pattern in relevant_patterns:
            all_rules.extend(pattern.rules)

        return all_rules
