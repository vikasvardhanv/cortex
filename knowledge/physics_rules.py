"""
Physics Rules for Cortex CEM

Contains fundamental physics equations and rules for:
- Heat transfer (conduction, convection, radiation)
- Structural mechanics (stress, strain, deformation)
- Fluid dynamics (basic flow equations)
- Thermodynamics

These rules are used by solvers and the LLM router to understand
what physics applies to a given problem.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import numpy as np
from enum import Enum


class PhysicsType(Enum):
    """Types of physics problems."""
    THERMAL = "thermal"
    STRUCTURAL = "structural"
    FLUID = "fluid"
    COUPLED_THERMAL_STRUCTURAL = "coupled_thermal_structural"
    COUPLED_FLUID_THERMAL = "coupled_fluid_thermal"
    MULTI_PHYSICS = "multi_physics"


@dataclass
class BoundaryCondition:
    """Represents a boundary condition for physics problems."""
    bc_type: str  # "dirichlet", "neumann", "convection", "radiation"
    location: str  # "left", "right", "top", "bottom", "all", or coordinates
    value: float  # Temperature, heat flux, etc.
    coefficient: Optional[float] = None  # For convection (h) or radiation (epsilon)

    def describe(self) -> str:
        if self.bc_type == "dirichlet":
            return f"Fixed value: {self.value} at {self.location}"
        elif self.bc_type == "neumann":
            return f"Heat flux: {self.value} W/m² at {self.location}"
        elif self.bc_type == "convection":
            return f"Convection: h={self.coefficient} W/(m²·K), T_inf={self.value}K at {self.location}"
        elif self.bc_type == "radiation":
            return f"Radiation: ε={self.coefficient}, T_inf={self.value}K at {self.location}"
        return f"{self.bc_type}: {self.value} at {self.location}"


class PhysicsRules:
    """
    Collection of physics equations and rules.

    Provides:
    - Governing equations for different physics types
    - Validation rules (physical constraints)
    - Unit conversions
    - Problem classification
    """

    # Physical constants
    STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
    BOLTZMANN = 1.380649e-23  # J/K
    UNIVERSAL_GAS = 8.314462618  # J/(mol·K)
    GRAVITY = 9.80665  # m/s²

    def __init__(self):
        self._rules: Dict[str, Dict] = {}
        self._load_default_rules()

    def _load_default_rules(self):
        """Load fundamental physics rules."""

        # === HEAT TRANSFER RULES ===
        self._rules["heat_conduction"] = {
            "name": "Fourier's Law of Heat Conduction",
            "equation": "q = -k * ∇T",
            "description": "Heat flux is proportional to temperature gradient",
            "governing_pde": "∂T/∂t = α∇²T (transient) or ∇²T = 0 (steady-state)",
            "parameters": ["thermal_conductivity", "temperature", "geometry"],
            "boundary_conditions": ["dirichlet", "neumann", "convection"],
            "physics_type": PhysicsType.THERMAL,
        }

        self._rules["heat_convection"] = {
            "name": "Newton's Law of Cooling",
            "equation": "q = h * (T_surface - T_fluid)",
            "description": "Convective heat transfer at a surface",
            "parameters": ["convection_coefficient", "surface_temp", "fluid_temp"],
            "physics_type": PhysicsType.THERMAL,
            "typical_h_values": {
                "natural_air": (5, 25),  # W/(m²·K)
                "forced_air": (25, 250),
                "natural_water": (100, 900),
                "forced_water": (300, 10000),
                "boiling_water": (2500, 25000),
            }
        }

        self._rules["heat_radiation"] = {
            "name": "Stefan-Boltzmann Law",
            "equation": "q = ε * σ * (T⁴ - T_surr⁴)",
            "description": "Radiative heat transfer between surfaces",
            "parameters": ["emissivity", "surface_temp", "surrounding_temp"],
            "physics_type": PhysicsType.THERMAL,
            "notes": "Significant at high temperatures (>500K)"
        }

        # === STRUCTURAL MECHANICS RULES ===
        self._rules["hookes_law"] = {
            "name": "Hooke's Law",
            "equation": "σ = E * ε",
            "description": "Stress is proportional to strain in elastic region",
            "parameters": ["youngs_modulus", "strain"],
            "physics_type": PhysicsType.STRUCTURAL,
        }

        self._rules["thermal_stress"] = {
            "name": "Thermal Stress",
            "equation": "σ_thermal = E * α * ΔT",
            "description": "Stress induced by temperature change in constrained material",
            "parameters": ["youngs_modulus", "thermal_expansion", "temperature_change"],
            "physics_type": PhysicsType.COUPLED_THERMAL_STRUCTURAL,
        }

        self._rules["yield_criterion"] = {
            "name": "Von Mises Yield Criterion",
            "equation": "σ_vm = √(σ₁² + σ₂² + σ₃² - σ₁σ₂ - σ₂σ₃ - σ₃σ₁) ≤ σ_yield",
            "description": "Material yields when von Mises stress exceeds yield strength",
            "parameters": ["principal_stresses", "yield_strength"],
            "physics_type": PhysicsType.STRUCTURAL,
        }

        # === FLUID DYNAMICS RULES ===
        self._rules["reynolds_number"] = {
            "name": "Reynolds Number",
            "equation": "Re = ρ * v * L / μ",
            "description": "Ratio of inertial to viscous forces, determines flow regime",
            "parameters": ["density", "velocity", "characteristic_length", "viscosity"],
            "physics_type": PhysicsType.FLUID,
            "regimes": {
                "laminar": (0, 2300),
                "transition": (2300, 4000),
                "turbulent": (4000, float('inf'))
            }
        }

        self._rules["bernoulli"] = {
            "name": "Bernoulli's Equation",
            "equation": "P + 0.5*ρ*v² + ρ*g*h = constant",
            "description": "Conservation of energy in fluid flow (inviscid, incompressible)",
            "parameters": ["pressure", "density", "velocity", "height"],
            "physics_type": PhysicsType.FLUID,
        }

    # === CALCULATION METHODS ===

    @staticmethod
    def heat_flux_conduction(k: float, dT: float, dx: float) -> float:
        """Calculate conductive heat flux (1D). Returns W/m²."""
        return -k * dT / dx

    @staticmethod
    def heat_flux_convection(h: float, T_surface: float, T_fluid: float) -> float:
        """Calculate convective heat flux. Returns W/m²."""
        return h * (T_surface - T_fluid)

    @staticmethod
    def heat_flux_radiation(epsilon: float, T_surface: float, T_surr: float) -> float:
        """Calculate radiative heat flux. Returns W/m²."""
        sigma = PhysicsRules.STEFAN_BOLTZMANN
        return epsilon * sigma * (T_surface**4 - T_surr**4)

    @staticmethod
    def thermal_diffusivity(k: float, rho: float, cp: float) -> float:
        """Calculate thermal diffusivity α = k/(ρ*cp). Returns m²/s."""
        return k / (rho * cp)

    @staticmethod
    def thermal_stress(E: float, alpha: float, delta_T: float) -> float:
        """Calculate thermal stress in fully constrained material. Returns Pa."""
        return E * alpha * delta_T

    @staticmethod
    def reynolds_number(rho: float, v: float, L: float, mu: float) -> float:
        """Calculate Reynolds number (dimensionless)."""
        return rho * v * L / mu

    @staticmethod
    def von_mises_stress(s1: float, s2: float, s3: float) -> float:
        """Calculate von Mises stress from principal stresses. Returns Pa."""
        return np.sqrt(0.5 * ((s1-s2)**2 + (s2-s3)**2 + (s3-s1)**2))

    # === VALIDATION METHODS ===

    def validate_temperature(self, T: float, material_name: str = None) -> Dict:
        """Validate if temperature is physically reasonable."""
        result = {"valid": True, "warnings": [], "errors": []}

        if T < 0:
            result["errors"].append("Temperature cannot be negative (in Kelvin)")
            result["valid"] = False
        elif T < 100:
            result["warnings"].append("Very low temperature - check if cryogenic analysis needed")
        elif T > 3000:
            result["warnings"].append("Extremely high temperature - limited materials available")

        return result

    def validate_stress(self, stress: float, yield_strength: float) -> Dict:
        """Validate stress against yield strength."""
        result = {"valid": True, "warnings": [], "errors": []}

        safety_factor = yield_strength / stress if stress > 0 else float('inf')

        if safety_factor < 1.0:
            result["errors"].append(f"FAILURE: Stress exceeds yield (SF={safety_factor:.2f})")
            result["valid"] = False
        elif safety_factor < 1.5:
            result["warnings"].append(f"Low safety factor: {safety_factor:.2f}")
        elif safety_factor < 2.0:
            result["warnings"].append(f"Moderate safety factor: {safety_factor:.2f}")

        result["safety_factor"] = safety_factor
        return result

    def classify_problem(self, keywords: List[str]) -> PhysicsType:
        """Classify physics problem type based on keywords."""
        keywords_lower = [k.lower() for k in keywords]

        has_thermal = any(k in keywords_lower for k in ["heat", "temperature", "thermal", "conduction", "hot", "cold"])
        has_structural = any(k in keywords_lower for k in ["stress", "strain", "force", "load", "structural", "deformation"])
        has_fluid = any(k in keywords_lower for k in ["flow", "fluid", "air", "pressure", "velocity", "cfd"])

        if has_thermal and has_structural:
            return PhysicsType.COUPLED_THERMAL_STRUCTURAL
        elif has_fluid and has_thermal:
            return PhysicsType.COUPLED_FLUID_THERMAL
        elif has_thermal:
            return PhysicsType.THERMAL
        elif has_structural:
            return PhysicsType.STRUCTURAL
        elif has_fluid:
            return PhysicsType.FLUID
        else:
            return PhysicsType.MULTI_PHYSICS

    def get_rule(self, name: str) -> Optional[Dict]:
        """Get a physics rule by name."""
        return self._rules.get(name)

    def list_rules(self) -> List[str]:
        """List all available rules."""
        return list(self._rules.keys())

    def rules_for_type(self, physics_type: PhysicsType) -> List[Dict]:
        """Get all rules applicable to a physics type."""
        return [r for r in self._rules.values() if r.get("physics_type") == physics_type]

    def describe_problem_setup(self, physics_type: PhysicsType) -> str:
        """Describe what's needed to set up a problem of given type."""
        setups = {
            PhysicsType.THERMAL: """
THERMAL ANALYSIS SETUP:
1. Geometry: Define the domain (plate, 3D object, etc.)
2. Material: Select material with thermal properties (k, ρ, cp)
3. Boundary Conditions:
   - Fixed temperature (Dirichlet): T = T₀ on boundary
   - Heat flux (Neumann): q = q₀ on boundary
   - Convection: q = h(T - T∞) on boundary
4. Initial Condition: T(x,0) = T_initial (for transient)
5. Solver: Steady-state (∇²T = 0) or Transient (∂T/∂t = α∇²T)
""",
            PhysicsType.STRUCTURAL: """
STRUCTURAL ANALYSIS SETUP:
1. Geometry: Define the domain
2. Material: Select material with mechanical properties (E, ν, σ_yield)
3. Boundary Conditions:
   - Fixed displacement: u = 0 on supports
   - Applied loads: F = F₀ on loaded surfaces
4. Solver: Static or Dynamic analysis
5. Post-process: Check von Mises stress < σ_yield
""",
            PhysicsType.COUPLED_THERMAL_STRUCTURAL: """
COUPLED THERMAL-STRUCTURAL SETUP:
1. First solve thermal problem → Temperature field T(x)
2. Calculate thermal strain: ε_thermal = α * (T - T_ref)
3. Solve structural problem with thermal strain as load
4. Check: σ_thermal + σ_mechanical < σ_yield
"""
        }
        return setups.get(physics_type, "Setup description not available for this physics type.")
