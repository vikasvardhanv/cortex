"""
Physics and Engineering Knowledge Base for Cortex CEM

Contains core equations, formulas, and physical laws with:
- Unit metadata for dimensional analysis
- Solver-type filters for direct integration
- Confidence/accuracy tags for sources
- Variable definitions and typical ranges
"""

from knowledge.rag.vector_db import CortexVectorDB


class PhysicsEngineeringKnowledgeIngestor:
    def __init__(self):
        self.db = None  # Injected by CortexIngestor

    def run(self):
        if self.db is None:
            self.db = CortexVectorDB()

        print("Ingesting Enhanced Engineering Knowledge Base...")

        knowledge_base = [
            # =====================================================================
            # ROCKET PROPULSION
            # =====================================================================
            {
                "title": "Tsiolkovsky Ideal Rocket Equation",
                "category": "propulsion",
                "solver_type": "orbital_mechanics",
                "content": "Delta-v = v_e * ln(m0 / mf). Where Delta-v is change in velocity, v_e is effective exhaust velocity (g0 * Isp), m0 is initial total mass (including propellant), and mf is final total mass (dry mass).",
                "equation_latex": r"\Delta v = v_e \ln\left(\frac{m_0}{m_f}\right)",
                "variables": {
                    "Delta_v": {"unit": "m/s", "description": "Change in velocity", "typical_range": "100-10000"},
                    "v_e": {"unit": "m/s", "description": "Effective exhaust velocity", "typical_range": "2000-4500"},
                    "m0": {"unit": "kg", "description": "Initial mass (wet)", "typical_range": "1-1e6"},
                    "mf": {"unit": "kg", "description": "Final mass (dry)", "typical_range": "0.1-1e5"}
                },
                "tags": ["rocket", "dynamics", "orbital", "equation", "fundamental"],
                "source": "Fundamentals of Astrodynamics - Bate, Mueller, White",
                "confidence": "high",
                "accuracy": "exact (ideal conditions)"
            },
            {
                "title": "Rocket Thrust Equation",
                "category": "propulsion",
                "solver_type": "fluid",
                "content": "F = m_dot * v_e + (p_e - p_a) * A_e. F is thrust, m_dot is mass flow rate, v_e is exit velocity, p_e is exit pressure, p_a is ambient pressure, A_e is exit area. First term is momentum thrust, second is pressure thrust.",
                "equation_latex": r"F = \dot{m} v_e + (p_e - p_a) A_e",
                "variables": {
                    "F": {"unit": "N", "description": "Thrust force", "typical_range": "1-1e7"},
                    "m_dot": {"unit": "kg/s", "description": "Mass flow rate", "typical_range": "0.1-1000"},
                    "v_e": {"unit": "m/s", "description": "Exit velocity", "typical_range": "2000-4500"},
                    "p_e": {"unit": "Pa", "description": "Exit pressure", "typical_range": "1e3-1e6"},
                    "p_a": {"unit": "Pa", "description": "Ambient pressure", "typical_range": "0-101325"},
                    "A_e": {"unit": "m^2", "description": "Exit area", "typical_range": "0.01-10"}
                },
                "tags": ["thrust", "nozzle", "equation", "fluid-dynamics", "fundamental"],
                "source": "NASA Rocket Propulsion Guide SP-8120",
                "confidence": "high",
                "accuracy": "exact (inviscid, 1D)"
            },
            {
                "title": "Specific Impulse (Isp) Definition",
                "category": "propulsion",
                "solver_type": "fluid",
                "content": "Isp = F / (m_dot * g0). It represents the thrust per unit weight flow of propellant. Higher Isp means more efficient propulsion. Isp has units of seconds.",
                "equation_latex": r"I_{sp} = \frac{F}{\dot{m} g_0} = \frac{v_e}{g_0}",
                "variables": {
                    "Isp": {"unit": "s", "description": "Specific impulse", "typical_range": "200-500"},
                    "F": {"unit": "N", "description": "Thrust", "typical_range": "1-1e7"},
                    "m_dot": {"unit": "kg/s", "description": "Mass flow rate", "typical_range": "0.1-1000"},
                    "g0": {"unit": "m/s^2", "description": "Standard gravity", "value": "9.80665"}
                },
                "tags": ["efficiency", "propulsion", "isp", "performance"],
                "source": "Space Mission Analysis and Design - Wertz & Larson",
                "confidence": "high",
                "accuracy": "exact"
            },
            {
                "title": "Characteristic Velocity (c*)",
                "category": "propulsion",
                "solver_type": "thermal",
                "content": "c* = (p_c * A_t) / m_dot. Characteristic velocity measures combustion efficiency independent of nozzle. It depends on combustion temperature, molecular weight, and specific heat ratio.",
                "equation_latex": r"c^* = \frac{p_c A_t}{\dot{m}} = \sqrt{\frac{R T_c}{\gamma}} \left(\frac{\gamma + 1}{2}\right)^{\frac{\gamma+1}{2(\gamma-1)}}",
                "variables": {
                    "c_star": {"unit": "m/s", "description": "Characteristic velocity", "typical_range": "1500-2500"},
                    "p_c": {"unit": "Pa", "description": "Chamber pressure", "typical_range": "1e6-30e6"},
                    "A_t": {"unit": "m^2", "description": "Throat area", "typical_range": "0.001-1"},
                    "T_c": {"unit": "K", "description": "Chamber temperature", "typical_range": "2500-3800"}
                },
                "tags": ["rocket", "combustion", "efficiency", "chamber"],
                "source": "Rocket Propulsion Elements - Sutton & Biblarz",
                "confidence": "high",
                "accuracy": "exact (ideal gas)"
            },
            {
                "title": "Nozzle Expansion Ratio",
                "category": "propulsion",
                "solver_type": "fluid",
                "content": "epsilon = A_e / A_t. The ratio of nozzle exit area to throat area. Optimal expansion ratio depends on ambient pressure. Over-expanded nozzle: p_e < p_a. Under-expanded: p_e > p_a.",
                "equation_latex": r"\epsilon = \frac{A_e}{A_t} = \frac{1}{M_e}\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M_e^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}",
                "variables": {
                    "epsilon": {"unit": "dimensionless", "description": "Expansion ratio", "typical_range": "5-100"},
                    "A_e": {"unit": "m^2", "description": "Exit area", "typical_range": "0.1-100"},
                    "A_t": {"unit": "m^2", "description": "Throat area", "typical_range": "0.01-10"},
                    "M_e": {"unit": "dimensionless", "description": "Exit Mach number", "typical_range": "2-5"}
                },
                "tags": ["nozzle", "geometry", "expansion", "supersonic"],
                "source": "Gas Dynamics - Anderson",
                "confidence": "high",
                "accuracy": "exact (isentropic)"
            },
            {
                "title": "Thrust Coefficient",
                "category": "propulsion",
                "solver_type": "fluid",
                "content": "C_F = F / (p_c * A_t). Dimensionless measure of nozzle performance. Depends on expansion ratio, pressure ratio, and specific heat ratio. Maximum C_F ~ 1.5-2.0 for typical nozzles.",
                "equation_latex": r"C_F = \frac{F}{p_c A_t} = \sqrt{\frac{2\gamma^2}{\gamma-1}\left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{\gamma-1}}\left[1-\left(\frac{p_e}{p_c}\right)^{\frac{\gamma-1}{\gamma}}\right]} + \frac{p_e - p_a}{p_c}\epsilon",
                "variables": {
                    "C_F": {"unit": "dimensionless", "description": "Thrust coefficient", "typical_range": "1.3-2.0"},
                    "F": {"unit": "N", "description": "Thrust", "typical_range": "1-1e7"},
                    "p_c": {"unit": "Pa", "description": "Chamber pressure", "typical_range": "1e6-30e6"},
                    "gamma": {"unit": "dimensionless", "description": "Specific heat ratio", "typical_range": "1.1-1.4"}
                },
                "tags": ["nozzle", "performance", "coefficient", "propulsion"],
                "source": "Rocket Propulsion Elements",
                "confidence": "high",
                "accuracy": "exact (isentropic)"
            },

            # =====================================================================
            # HEAT TRANSFER
            # =====================================================================
            {
                "title": "Fourier's Law of Heat Conduction",
                "category": "thermal",
                "solver_type": "thermal",
                "content": "q = -k * grad(T). Heat flux (q) is proportional to the negative temperature gradient. k is thermal conductivity. Negative sign indicates heat flows from hot to cold.",
                "equation_latex": r"\vec{q} = -k \nabla T",
                "variables": {
                    "q": {"unit": "W/m^2", "description": "Heat flux", "typical_range": "1e2-1e7"},
                    "k": {"unit": "W/(m*K)", "description": "Thermal conductivity", "typical_range": "0.1-400"},
                    "T": {"unit": "K", "description": "Temperature", "typical_range": "200-3000"},
                    "grad_T": {"unit": "K/m", "description": "Temperature gradient", "typical_range": "1-1e6"}
                },
                "tags": ["conduction", "thermal", "physics", "equation", "fundamental"],
                "source": "Incropera & DeWitt - Fundamentals of Heat and Mass Transfer",
                "confidence": "high",
                "accuracy": "exact (isotropic material)"
            },
            {
                "title": "Stefan-Boltzmann Law (Radiation)",
                "category": "thermal",
                "solver_type": "thermal",
                "content": "P = epsilon * sigma * A * T^4. P is radiated power, epsilon is emissivity (0-1), sigma is Stefan-Boltzmann constant (5.67e-8 W/m^2K^4), A is surface area, T is absolute temperature.",
                "equation_latex": r"P = \varepsilon \sigma A T^4",
                "variables": {
                    "P": {"unit": "W", "description": "Radiated power", "typical_range": "1-1e6"},
                    "epsilon": {"unit": "dimensionless", "description": "Emissivity", "typical_range": "0.1-1.0"},
                    "sigma": {"unit": "W/(m^2*K^4)", "description": "Stefan-Boltzmann constant", "value": "5.670374419e-8"},
                    "A": {"unit": "m^2", "description": "Surface area", "typical_range": "0.01-100"},
                    "T": {"unit": "K", "description": "Absolute temperature", "typical_range": "200-5000"}
                },
                "tags": ["radiation", "thermal", "physics", "equation", "fundamental"],
                "source": "Physics Handbook - Boltzmann, Stefan",
                "confidence": "high",
                "accuracy": "exact"
            },
            {
                "title": "Newton's Law of Cooling (Convection)",
                "category": "thermal",
                "solver_type": "thermal",
                "content": "q = h * (T_s - T_inf). q is heat flux, h is convection heat transfer coefficient, T_s is surface temperature, T_inf is fluid temperature. h depends on flow conditions and geometry.",
                "equation_latex": r"q = h (T_s - T_\infty)",
                "variables": {
                    "q": {"unit": "W/m^2", "description": "Heat flux", "typical_range": "1e2-1e6"},
                    "h": {"unit": "W/(m^2*K)", "description": "Heat transfer coefficient", "typical_range": "5-50000"},
                    "T_s": {"unit": "K", "description": "Surface temperature", "typical_range": "200-2000"},
                    "T_inf": {"unit": "K", "description": "Fluid temperature", "typical_range": "200-2000"}
                },
                "tags": ["convection", "thermal", "fluid", "equation", "fundamental"],
                "source": "Engineering ToolBox / Heat Transfer Texts",
                "confidence": "high",
                "accuracy": "phenomenological"
            },
            {
                "title": "Heat Equation (Transient Conduction)",
                "category": "thermal",
                "solver_type": "thermal",
                "content": "rho * cp * dT/dt = k * laplacian(T) + Q. Time-dependent heat conduction with volumetric heat generation Q. alpha = k/(rho*cp) is thermal diffusivity.",
                "equation_latex": r"\rho c_p \frac{\partial T}{\partial t} = k \nabla^2 T + \dot{Q}",
                "variables": {
                    "rho": {"unit": "kg/m^3", "description": "Density", "typical_range": "1000-20000"},
                    "cp": {"unit": "J/(kg*K)", "description": "Specific heat", "typical_range": "100-4200"},
                    "k": {"unit": "W/(m*K)", "description": "Thermal conductivity", "typical_range": "0.1-400"},
                    "Q": {"unit": "W/m^3", "description": "Volumetric heat generation", "typical_range": "0-1e9"},
                    "alpha": {"unit": "m^2/s", "description": "Thermal diffusivity", "typical_range": "1e-7-1e-4"}
                },
                "tags": ["conduction", "transient", "PDE", "thermal"],
                "source": "Incropera & DeWitt",
                "confidence": "high",
                "accuracy": "exact"
            },
            {
                "title": "Biot Number",
                "category": "thermal",
                "solver_type": "thermal",
                "content": "Bi = h * L_c / k. Ratio of convective to conductive heat transfer. Bi < 0.1: lumped capacitance valid. Bi > 0.1: temperature gradients significant inside body.",
                "equation_latex": r"Bi = \frac{h L_c}{k}",
                "variables": {
                    "Bi": {"unit": "dimensionless", "description": "Biot number", "typical_range": "0.001-100"},
                    "h": {"unit": "W/(m^2*K)", "description": "Heat transfer coefficient", "typical_range": "5-50000"},
                    "L_c": {"unit": "m", "description": "Characteristic length (V/A)", "typical_range": "0.001-1"},
                    "k": {"unit": "W/(m*K)", "description": "Thermal conductivity", "typical_range": "0.1-400"}
                },
                "tags": ["dimensionless", "thermal", "convection", "conduction"],
                "source": "Heat Transfer Fundamentals",
                "confidence": "high",
                "accuracy": "exact"
            },
            {
                "title": "Nusselt Number Correlation (Forced Convection)",
                "category": "thermal",
                "solver_type": "thermal",
                "content": "Nu = h * L / k_f = C * Re^m * Pr^n. Dimensionless heat transfer coefficient. For turbulent pipe flow: Nu = 0.023 * Re^0.8 * Pr^0.4 (Dittus-Boelter).",
                "equation_latex": r"Nu = \frac{hL}{k_f} = C \cdot Re^m \cdot Pr^n",
                "variables": {
                    "Nu": {"unit": "dimensionless", "description": "Nusselt number", "typical_range": "1-1000"},
                    "h": {"unit": "W/(m^2*K)", "description": "Heat transfer coefficient", "typical_range": "5-50000"},
                    "Re": {"unit": "dimensionless", "description": "Reynolds number", "typical_range": "1-1e7"},
                    "Pr": {"unit": "dimensionless", "description": "Prandtl number", "typical_range": "0.01-1000"}
                },
                "tags": ["convection", "correlation", "dimensionless", "heat-transfer"],
                "source": "Incropera & DeWitt",
                "confidence": "medium",
                "accuracy": "empirical (±15-25%)"
            },

            # =====================================================================
            # STRUCTURAL MECHANICS
            # =====================================================================
            {
                "title": "Hooke's Law (Elasticity)",
                "category": "structural",
                "solver_type": "structural",
                "content": "sigma = E * epsilon. Stress equals Young's Modulus times strain. Valid in linear elastic region. For shear: tau = G * gamma where G = E / (2*(1+nu)).",
                "equation_latex": r"\sigma = E \varepsilon",
                "variables": {
                    "sigma": {"unit": "Pa", "description": "Normal stress", "typical_range": "1e6-1e9"},
                    "E": {"unit": "Pa", "description": "Young's modulus", "typical_range": "1e9-500e9"},
                    "epsilon": {"unit": "dimensionless", "description": "Strain", "typical_range": "0-0.01"},
                    "G": {"unit": "Pa", "description": "Shear modulus", "typical_range": "1e9-200e9"},
                    "nu": {"unit": "dimensionless", "description": "Poisson's ratio", "typical_range": "0.2-0.5"}
                },
                "tags": ["mechanics", "stress", "strain", "material", "fundamental"],
                "source": "Mechanics of Materials - Hibbeler",
                "confidence": "high",
                "accuracy": "exact (linear elastic)"
            },
            {
                "title": "Generalized Hooke's Law (3D)",
                "category": "structural",
                "solver_type": "structural",
                "content": "epsilon_ij = (1+nu)/E * sigma_ij - nu/E * delta_ij * sigma_kk. Full 3D constitutive relation for isotropic linear elasticity. Relates strain tensor to stress tensor.",
                "equation_latex": r"\varepsilon_{ij} = \frac{1+\nu}{E}\sigma_{ij} - \frac{\nu}{E}\delta_{ij}\sigma_{kk}",
                "variables": {
                    "epsilon_ij": {"unit": "dimensionless", "description": "Strain tensor", "typical_range": "0-0.01"},
                    "sigma_ij": {"unit": "Pa", "description": "Stress tensor", "typical_range": "1e6-1e9"},
                    "E": {"unit": "Pa", "description": "Young's modulus", "typical_range": "1e9-500e9"},
                    "nu": {"unit": "dimensionless", "description": "Poisson's ratio", "typical_range": "0.2-0.5"}
                },
                "tags": ["elasticity", "tensor", "constitutive", "3D"],
                "source": "Theory of Elasticity - Timoshenko",
                "confidence": "high",
                "accuracy": "exact (linear elastic isotropic)"
            },
            {
                "title": "Euler-Bernoulli Beam Equation",
                "category": "structural",
                "solver_type": "structural",
                "content": "EI * (d^4w/dx^4) = q(x). E is Young's Modulus, I is area moment of inertia, w is transverse deflection, q is distributed load. Assumes small deflections and slender beam.",
                "equation_latex": r"EI \frac{d^4 w}{dx^4} = q(x)",
                "variables": {
                    "E": {"unit": "Pa", "description": "Young's modulus", "typical_range": "1e9-500e9"},
                    "I": {"unit": "m^4", "description": "Area moment of inertia", "typical_range": "1e-12-1e-3"},
                    "w": {"unit": "m", "description": "Deflection", "typical_range": "0-0.1"},
                    "q": {"unit": "N/m", "description": "Distributed load", "typical_range": "1-1e6"}
                },
                "tags": ["beam", "deflection", "statics", "equation", "fundamental"],
                "source": "Structural Design Standards / Roark's Formulas",
                "confidence": "high",
                "accuracy": "exact (small deflections)"
            },
            {
                "title": "Von Mises Yield Criterion",
                "category": "structural",
                "solver_type": "structural",
                "content": "sigma_vm = sqrt(0.5*((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2)). Material yields when von Mises stress equals yield strength. Most common criterion for ductile metals.",
                "equation_latex": r"\sigma_{vm} = \sqrt{\frac{1}{2}\left[(\sigma_1-\sigma_2)^2 + (\sigma_2-\sigma_3)^2 + (\sigma_3-\sigma_1)^2\right]}",
                "variables": {
                    "sigma_vm": {"unit": "Pa", "description": "Von Mises equivalent stress", "typical_range": "1e6-1e9"},
                    "sigma_1": {"unit": "Pa", "description": "Principal stress 1", "typical_range": "-1e9 to 1e9"},
                    "sigma_2": {"unit": "Pa", "description": "Principal stress 2", "typical_range": "-1e9 to 1e9"},
                    "sigma_3": {"unit": "Pa", "description": "Principal stress 3", "typical_range": "-1e9 to 1e9"},
                    "sigma_y": {"unit": "Pa", "description": "Yield strength", "typical_range": "100e6-2000e6"}
                },
                "tags": ["failure", "yield", "stress", "mechanical-engineering", "ductile"],
                "source": "Shigley's Mechanical Engineering Design",
                "confidence": "high",
                "accuracy": "exact (isotropic ductile)"
            },
            {
                "title": "Thermal Stress",
                "category": "structural",
                "solver_type": "structural",
                "content": "sigma_th = E * alpha * Delta_T. Stress induced by constrained thermal expansion. alpha is coefficient of thermal expansion. For biaxial constraint: sigma = E*alpha*DT/(1-nu).",
                "equation_latex": r"\sigma_{th} = E \alpha \Delta T",
                "variables": {
                    "sigma_th": {"unit": "Pa", "description": "Thermal stress", "typical_range": "1e6-1e9"},
                    "E": {"unit": "Pa", "description": "Young's modulus", "typical_range": "1e9-500e9"},
                    "alpha": {"unit": "1/K", "description": "Thermal expansion coefficient", "typical_range": "1e-6-30e-6"},
                    "Delta_T": {"unit": "K", "description": "Temperature change", "typical_range": "10-1000"}
                },
                "tags": ["thermal", "stress", "expansion", "coupled"],
                "source": "Mechanics of Materials",
                "confidence": "high",
                "accuracy": "exact (linear elastic)"
            },
            {
                "title": "FEM Stiffness Matrix",
                "category": "structural",
                "solver_type": "structural",
                "content": "[K] = Integral_V([B]^T [D] [B] dV). K is element stiffness matrix, B is strain-displacement matrix, D is material constitutive matrix. Fundamental equation of finite element method.",
                "equation_latex": r"[K] = \int_V [B]^T [D] [B] \, dV",
                "variables": {
                    "K": {"unit": "N/m", "description": "Stiffness matrix", "typical_range": "1e6-1e12"},
                    "B": {"unit": "1/m", "description": "Strain-displacement matrix", "typical_range": "varies"},
                    "D": {"unit": "Pa", "description": "Constitutive matrix", "typical_range": "1e9-1e12"}
                },
                "tags": ["FEM", "structural", "stiffness-matrix", "numerical"],
                "source": "Finite Element Analysis - Bathe",
                "confidence": "high",
                "accuracy": "exact (within FEM assumptions)"
            },

            # =====================================================================
            # FLUID DYNAMICS
            # =====================================================================
            {
                "title": "Reynolds Number (Re)",
                "category": "fluid",
                "solver_type": "fluid",
                "content": "Re = (rho * v * L) / mu = v * L / nu. Ratio of inertial to viscous forces. Re < 2300: laminar (pipe). Re > 4000: turbulent. Critical for flow regime prediction.",
                "equation_latex": r"Re = \frac{\rho v L}{\mu} = \frac{v L}{\nu}",
                "variables": {
                    "Re": {"unit": "dimensionless", "description": "Reynolds number", "typical_range": "0.01-1e8"},
                    "rho": {"unit": "kg/m^3", "description": "Fluid density", "typical_range": "0.1-13000"},
                    "v": {"unit": "m/s", "description": "Flow velocity", "typical_range": "0.01-1000"},
                    "L": {"unit": "m", "description": "Characteristic length", "typical_range": "0.001-100"},
                    "mu": {"unit": "Pa*s", "description": "Dynamic viscosity", "typical_range": "1e-5-100"},
                    "nu": {"unit": "m^2/s", "description": "Kinematic viscosity", "typical_range": "1e-7-1e-3"}
                },
                "tags": ["fluids", "dimensionless", "turbulence", "fundamental"],
                "source": "Fluid Mechanics - White",
                "confidence": "high",
                "accuracy": "exact"
            },
            {
                "title": "Mach Number (Ma)",
                "category": "fluid",
                "solver_type": "fluid",
                "content": "Ma = v / a. Ratio of flow velocity to local speed of sound. a = sqrt(gamma * R * T) for ideal gas. Ma < 0.3: incompressible. Ma < 1: subsonic. Ma > 1: supersonic.",
                "equation_latex": r"Ma = \frac{v}{a} = \frac{v}{\sqrt{\gamma R T}}",
                "variables": {
                    "Ma": {"unit": "dimensionless", "description": "Mach number", "typical_range": "0-10"},
                    "v": {"unit": "m/s", "description": "Flow velocity", "typical_range": "0-3000"},
                    "a": {"unit": "m/s", "description": "Speed of sound", "typical_range": "200-400"},
                    "gamma": {"unit": "dimensionless", "description": "Specific heat ratio", "typical_range": "1.1-1.67"},
                    "T": {"unit": "K", "description": "Temperature", "typical_range": "200-3000"}
                },
                "tags": ["aerodynamics", "supersonic", "speed-of-sound", "compressible"],
                "source": "Modern Compressible Flow - Anderson",
                "confidence": "high",
                "accuracy": "exact"
            },
            {
                "title": "Prandtl Number (Pr)",
                "category": "fluid",
                "solver_type": "thermal",
                "content": "Pr = nu / alpha = (mu * cp) / k. Ratio of momentum diffusivity to thermal diffusivity. Pr < 1: thermal diffuses faster. Pr > 1: momentum diffuses faster. Air: Pr ~ 0.7.",
                "equation_latex": r"Pr = \frac{\nu}{\alpha} = \frac{\mu c_p}{k}",
                "variables": {
                    "Pr": {"unit": "dimensionless", "description": "Prandtl number", "typical_range": "0.01-1000"},
                    "nu": {"unit": "m^2/s", "description": "Kinematic viscosity", "typical_range": "1e-7-1e-3"},
                    "alpha": {"unit": "m^2/s", "description": "Thermal diffusivity", "typical_range": "1e-7-1e-4"},
                    "cp": {"unit": "J/(kg*K)", "description": "Specific heat", "typical_range": "100-4200"},
                    "k": {"unit": "W/(m*K)", "description": "Thermal conductivity", "typical_range": "0.01-400"}
                },
                "tags": ["heat-transfer", "fluids", "dimensionless", "convection"],
                "source": "Heat Transfer - Incropera",
                "confidence": "high",
                "accuracy": "exact"
            },
            {
                "title": "Navier-Stokes Momentum Equation",
                "category": "fluid",
                "solver_type": "fluid",
                "content": "rho * (Dv/Dt) = -grad(p) + mu * laplacian(v) + rho * g. Conservation of momentum for viscous fluid. D/Dt is material derivative. Forms basis of CFD.",
                "equation_latex": r"\rho \frac{D\vec{v}}{Dt} = -\nabla p + \mu \nabla^2 \vec{v} + \rho \vec{g}",
                "variables": {
                    "rho": {"unit": "kg/m^3", "description": "Density", "typical_range": "0.1-13000"},
                    "v": {"unit": "m/s", "description": "Velocity vector", "typical_range": "0-1000"},
                    "p": {"unit": "Pa", "description": "Pressure", "typical_range": "1e3-1e8"},
                    "mu": {"unit": "Pa*s", "description": "Dynamic viscosity", "typical_range": "1e-5-100"},
                    "g": {"unit": "m/s^2", "description": "Gravity", "value": "9.81"}
                },
                "tags": ["CFD", "fundamental", "PDE", "viscous"],
                "source": "Computational Fluid Dynamics - Anderson",
                "confidence": "high",
                "accuracy": "exact (continuum)"
            },
            {
                "title": "Bernoulli Equation",
                "category": "fluid",
                "solver_type": "fluid",
                "content": "p + 0.5*rho*v^2 + rho*g*z = constant. Conservation of energy along streamline for inviscid, incompressible, steady flow. Sum of static, dynamic, and hydrostatic pressure.",
                "equation_latex": r"p + \frac{1}{2}\rho v^2 + \rho g z = \text{constant}",
                "variables": {
                    "p": {"unit": "Pa", "description": "Static pressure", "typical_range": "1e3-1e8"},
                    "rho": {"unit": "kg/m^3", "description": "Density", "typical_range": "0.1-13000"},
                    "v": {"unit": "m/s", "description": "Velocity", "typical_range": "0-100"},
                    "g": {"unit": "m/s^2", "description": "Gravity", "value": "9.81"},
                    "z": {"unit": "m", "description": "Height", "typical_range": "0-1000"}
                },
                "tags": ["inviscid", "energy", "streamline", "fundamental"],
                "source": "Fluid Mechanics",
                "confidence": "high",
                "accuracy": "exact (inviscid, incompressible)"
            },
            {
                "title": "Darcy-Weisbach Equation",
                "category": "fluid",
                "solver_type": "fluid",
                "content": "Delta_p = f * (L/D) * (rho*v^2/2). Pressure drop in pipe flow. f is Darcy friction factor (from Moody chart or Colebrook equation).",
                "equation_latex": r"\Delta p = f \frac{L}{D} \frac{\rho v^2}{2}",
                "variables": {
                    "Delta_p": {"unit": "Pa", "description": "Pressure drop", "typical_range": "1-1e6"},
                    "f": {"unit": "dimensionless", "description": "Friction factor", "typical_range": "0.01-0.1"},
                    "L": {"unit": "m", "description": "Pipe length", "typical_range": "1-10000"},
                    "D": {"unit": "m", "description": "Pipe diameter", "typical_range": "0.01-10"},
                    "v": {"unit": "m/s", "description": "Velocity", "typical_range": "0.1-30"}
                },
                "tags": ["pipe-flow", "friction", "pressure-drop", "hydraulics"],
                "source": "Fluid Mechanics - White",
                "confidence": "high",
                "accuracy": "empirical (well-validated)"
            },

            # =====================================================================
            # THERMODYNAMICS
            # =====================================================================
            {
                "title": "First Law of Thermodynamics",
                "category": "thermodynamics",
                "solver_type": "thermal",
                "content": "Delta_U = Q - W. Change in internal energy equals heat added minus work done by system. For closed system. Can also write dU = delta_Q - delta_W.",
                "equation_latex": r"\Delta U = Q - W",
                "variables": {
                    "Delta_U": {"unit": "J", "description": "Change in internal energy", "typical_range": "1-1e9"},
                    "Q": {"unit": "J", "description": "Heat added", "typical_range": "1-1e9"},
                    "W": {"unit": "J", "description": "Work done by system", "typical_range": "1-1e9"}
                },
                "tags": ["energy", "conservation", "thermodynamics", "fundamental"],
                "source": "Fundamentals of Engineering Thermodynamics - Moran",
                "confidence": "high",
                "accuracy": "exact (law)"
            },
            {
                "title": "Ideal Gas Law",
                "category": "thermodynamics",
                "solver_type": "thermal",
                "content": "PV = nRT = mRT/M. Equation of state for ideal gas. R = 8.314 J/(mol*K) universal. For air: R_specific = 287 J/(kg*K).",
                "equation_latex": r"PV = nRT",
                "variables": {
                    "P": {"unit": "Pa", "description": "Pressure", "typical_range": "1e3-1e8"},
                    "V": {"unit": "m^3", "description": "Volume", "typical_range": "1e-6-1e6"},
                    "n": {"unit": "mol", "description": "Amount of substance", "typical_range": "0.01-1e6"},
                    "R": {"unit": "J/(mol*K)", "description": "Universal gas constant", "value": "8.314"},
                    "T": {"unit": "K", "description": "Temperature", "typical_range": "100-5000"}
                },
                "tags": ["gas", "pressure", "standard-physics", "equation-of-state"],
                "source": "Physics Handbook",
                "confidence": "high",
                "accuracy": "good (ideal gas assumption)"
            },
            {
                "title": "Entropy Definition (Second Law)",
                "category": "thermodynamics",
                "solver_type": "thermal",
                "content": "dS >= delta_Q / T. Entropy change >= heat/temperature. Equality for reversible processes. Entropy always increases in isolated system.",
                "equation_latex": r"dS \geq \frac{\delta Q}{T}",
                "variables": {
                    "S": {"unit": "J/K", "description": "Entropy", "typical_range": "1-1e6"},
                    "Q": {"unit": "J", "description": "Heat", "typical_range": "1-1e9"},
                    "T": {"unit": "K", "description": "Temperature", "typical_range": "100-5000"}
                },
                "tags": ["entropy", "energy", "efficiency", "fundamental"],
                "source": "Thermodynamics Core",
                "confidence": "high",
                "accuracy": "exact (law)"
            },
            {
                "title": "Carnot Efficiency",
                "category": "thermodynamics",
                "solver_type": "thermal",
                "content": "eta_carnot = 1 - T_cold / T_hot. Maximum theoretical efficiency of heat engine. Real engines always have lower efficiency due to irreversibilities.",
                "equation_latex": r"\eta_{Carnot} = 1 - \frac{T_C}{T_H}",
                "variables": {
                    "eta": {"unit": "dimensionless", "description": "Efficiency", "typical_range": "0.1-0.8"},
                    "T_cold": {"unit": "K", "description": "Cold reservoir temperature", "typical_range": "200-400"},
                    "T_hot": {"unit": "K", "description": "Hot reservoir temperature", "typical_range": "300-3000"}
                },
                "tags": ["engine", "efficiency", "cycle", "maximum"],
                "source": "Engineering Science",
                "confidence": "high",
                "accuracy": "exact (theoretical maximum)"
            },
            {
                "title": "Isentropic Flow Relations",
                "category": "thermodynamics",
                "solver_type": "fluid",
                "content": "T/T0 = (1 + (gamma-1)/2 * M^2)^(-1). Temperature ratio for isentropic flow. T0 is stagnation temperature. Similar relations exist for pressure and density.",
                "equation_latex": r"\frac{T}{T_0} = \left(1 + \frac{\gamma-1}{2}M^2\right)^{-1}",
                "variables": {
                    "T": {"unit": "K", "description": "Static temperature", "typical_range": "100-3000"},
                    "T0": {"unit": "K", "description": "Stagnation temperature", "typical_range": "200-4000"},
                    "M": {"unit": "dimensionless", "description": "Mach number", "typical_range": "0-5"},
                    "gamma": {"unit": "dimensionless", "description": "Specific heat ratio", "typical_range": "1.1-1.67"}
                },
                "tags": ["compressible", "isentropic", "supersonic", "gas-dynamics"],
                "source": "Modern Compressible Flow - Anderson",
                "confidence": "high",
                "accuracy": "exact (isentropic)"
            },

            # =====================================================================
            # ORBITAL MECHANICS
            # =====================================================================
            {
                "title": "Kepler's Third Law",
                "category": "orbital",
                "solver_type": "orbital_mechanics",
                "content": "T^2 = (4*pi^2 / GM) * a^3. Period squared proportional to semi-major axis cubed. For Earth: mu = GM = 3.986e14 m^3/s^2.",
                "equation_latex": r"T^2 = \frac{4\pi^2}{GM}a^3",
                "variables": {
                    "T": {"unit": "s", "description": "Orbital period", "typical_range": "5400-3.15e7"},
                    "a": {"unit": "m", "description": "Semi-major axis", "typical_range": "6.5e6-3.84e8"},
                    "G": {"unit": "m^3/(kg*s^2)", "description": "Gravitational constant", "value": "6.674e-11"},
                    "M": {"unit": "kg", "description": "Central body mass", "typical_range": "5.97e24-1.99e30"}
                },
                "tags": ["orbit", "kepler", "planetary", "period"],
                "source": "Classical Mechanics - Goldstein",
                "confidence": "high",
                "accuracy": "exact (two-body)"
            },
            {
                "title": "Escape Velocity",
                "category": "orbital",
                "solver_type": "orbital_mechanics",
                "content": "v_esc = sqrt(2 * G * M / r) = sqrt(2 * mu / r). Minimum velocity to escape gravitational field. For Earth surface: v_esc = 11.2 km/s.",
                "equation_latex": r"v_{esc} = \sqrt{\frac{2GM}{r}} = \sqrt{\frac{2\mu}{r}}",
                "variables": {
                    "v_esc": {"unit": "m/s", "description": "Escape velocity", "typical_range": "1000-620000"},
                    "G": {"unit": "m^3/(kg*s^2)", "description": "Gravitational constant", "value": "6.674e-11"},
                    "M": {"unit": "kg", "description": "Body mass", "typical_range": "1e20-2e30"},
                    "r": {"unit": "m", "description": "Distance from center", "typical_range": "1e6-1e11"},
                    "mu": {"unit": "m^3/s^2", "description": "Standard gravitational parameter", "typical_range": "1e10-1.3e20"}
                },
                "tags": ["gravity", "escape", "velocity", "orbital"],
                "source": "Fundamentals of Astrodynamics",
                "confidence": "high",
                "accuracy": "exact"
            },
            {
                "title": "Vis-Viva Equation",
                "category": "orbital",
                "solver_type": "orbital_mechanics",
                "content": "v^2 = mu * (2/r - 1/a). Orbital velocity at any point. a is semi-major axis, r is current radius. For circular orbit (r=a): v = sqrt(mu/r).",
                "equation_latex": r"v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right)",
                "variables": {
                    "v": {"unit": "m/s", "description": "Orbital velocity", "typical_range": "1000-50000"},
                    "mu": {"unit": "m^3/s^2", "description": "Gravitational parameter", "typical_range": "1e10-1.3e20"},
                    "r": {"unit": "m", "description": "Current radius", "typical_range": "6.5e6-1e11"},
                    "a": {"unit": "m", "description": "Semi-major axis", "typical_range": "6.5e6-1e11"}
                },
                "tags": ["orbital", "velocity", "energy", "trajectory"],
                "source": "Orbital Mechanics - Curtis",
                "confidence": "high",
                "accuracy": "exact (two-body)"
            },

            # =====================================================================
            # LATTICE BOLTZMANN / CFD
            # =====================================================================
            {
                "title": "LBM Equilibrium Distribution (BGK)",
                "category": "fluid",
                "solver_type": "fluid",
                "content": "f_eq = w_i * rho * (1 + (c_i . u)/cs^2 + (c_i . u)^2/(2*cs^4) - u^2/(2*cs^2)). Equilibrium distribution function for D2Q9/D3Q19 lattice. cs^2 = 1/3.",
                "equation_latex": r"f_i^{eq} = w_i \rho \left[1 + \frac{\vec{c}_i \cdot \vec{u}}{c_s^2} + \frac{(\vec{c}_i \cdot \vec{u})^2}{2c_s^4} - \frac{u^2}{2c_s^2}\right]",
                "variables": {
                    "f_eq": {"unit": "dimensionless", "description": "Equilibrium distribution", "typical_range": "0-1"},
                    "w_i": {"unit": "dimensionless", "description": "Lattice weight", "typical_range": "0-0.5"},
                    "rho": {"unit": "dimensionless", "description": "Density (lattice)", "typical_range": "0.5-2"},
                    "u": {"unit": "dimensionless", "description": "Velocity (lattice)", "typical_range": "0-0.3"},
                    "cs": {"unit": "dimensionless", "description": "Lattice speed of sound", "value": "1/sqrt(3)"}
                },
                "tags": ["LBM", "lattice-boltzmann", "CFD", "mesoscopic"],
                "source": "The Lattice Boltzmann Method - Kruger et al.",
                "confidence": "high",
                "accuracy": "exact (low Ma)"
            },
            {
                "title": "LBM Relaxation and Viscosity",
                "category": "fluid",
                "solver_type": "fluid",
                "content": "nu = cs^2 * (tau - 0.5) * dt. Kinematic viscosity in LBM from relaxation time. tau > 0.5 required for stability. tau close to 0.5 gives low viscosity.",
                "equation_latex": r"\nu = c_s^2 \left(\tau - \frac{1}{2}\right) \Delta t",
                "variables": {
                    "nu": {"unit": "m^2/s", "description": "Kinematic viscosity", "typical_range": "1e-7-1e-3"},
                    "tau": {"unit": "dimensionless", "description": "Relaxation time", "typical_range": "0.51-2"},
                    "cs": {"unit": "dimensionless", "description": "Lattice speed of sound", "value": "1/sqrt(3)"},
                    "dt": {"unit": "s", "description": "Time step", "typical_range": "1e-6-1e-3"}
                },
                "tags": ["LBM", "viscosity", "relaxation", "stability"],
                "source": "LBM Fundamentals",
                "confidence": "high",
                "accuracy": "exact"
            },

            # =====================================================================
            # MATHEMATICS / NUMERICAL METHODS
            # =====================================================================
            {
                "title": "Divergence Theorem (Gauss's Theorem)",
                "category": "mathematics",
                "solver_type": "general",
                "content": "Integral_V(div F dV) = Integral_S(F . n dS). Volume integral of divergence equals surface integral of flux. Fundamental for conservation laws.",
                "equation_latex": r"\int_V \nabla \cdot \vec{F} \, dV = \oint_S \vec{F} \cdot \hat{n} \, dS",
                "variables": {
                    "F": {"unit": "varies", "description": "Vector field", "typical_range": "varies"},
                    "V": {"unit": "m^3", "description": "Volume", "typical_range": "varies"},
                    "S": {"unit": "m^2", "description": "Surface area", "typical_range": "varies"},
                    "n": {"unit": "dimensionless", "description": "Unit normal", "typical_range": "unit vector"}
                },
                "tags": ["calculus", "vector-field", "flux", "theorem"],
                "source": "Vector Calculus - Marsden & Tromba",
                "confidence": "high",
                "accuracy": "exact (mathematical theorem)"
            },
            {
                "title": "Taylor Series Expansion",
                "category": "mathematics",
                "solver_type": "general",
                "content": "f(x+h) = f(x) + h*f'(x) + h^2/2!*f''(x) + O(h^3). Basis for finite difference approximations. Second-order central difference: f'(x) ≈ (f(x+h) - f(x-h))/(2h).",
                "equation_latex": r"f(x+h) = f(x) + hf'(x) + \frac{h^2}{2!}f''(x) + \mathcal{O}(h^3)",
                "variables": {
                    "f": {"unit": "varies", "description": "Function", "typical_range": "varies"},
                    "h": {"unit": "varies", "description": "Step size", "typical_range": "1e-6-1"},
                    "x": {"unit": "varies", "description": "Variable", "typical_range": "varies"}
                },
                "tags": ["numerical", "approximation", "finite-difference", "FDM"],
                "source": "Numerical Analysis - Burden & Faires",
                "confidence": "high",
                "accuracy": "exact (mathematical)"
            },
            {
                "title": "Gauss Quadrature (Numerical Integration)",
                "category": "mathematics",
                "solver_type": "structural",
                "content": "Integral_(-1)^1 f(x) dx ≈ Sum(w_i * f(x_i)). Optimal numerical integration using Gauss points. n-point rule exact for polynomials up to degree 2n-1.",
                "equation_latex": r"\int_{-1}^{1} f(x) \, dx \approx \sum_{i=1}^{n} w_i f(x_i)",
                "variables": {
                    "w_i": {"unit": "dimensionless", "description": "Quadrature weights", "typical_range": "0-2"},
                    "x_i": {"unit": "dimensionless", "description": "Quadrature points", "typical_range": "-1 to 1"},
                    "n": {"unit": "integer", "description": "Number of points", "typical_range": "1-10"}
                },
                "tags": ["numerical", "integration", "FEM", "quadrature"],
                "source": "Numerical Methods for Engineers",
                "confidence": "high",
                "accuracy": "exact (for polynomial degree)"
            }
        ]

        print(f"Batch uploading {len(knowledge_base)} enhanced engineering entries...")
        self.db.upsert_knowledge_batch(knowledge_base)
        print("Enhanced physics knowledge ingested successfully.")


if __name__ == "__main__":
    ingestor = PhysicsEngineeringKnowledgeIngestor()
    ingestor.run()
