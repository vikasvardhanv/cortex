"""
Router for Cortex CEM

Uses Claude API to:
1. Parse natural language problem descriptions
2. Extract engineering parameters
3. Determine which solvers are needed
4. Generate problem specifications

Example:
    router = Router()
    spec = router.parse("Design a heat shield for Mars re-entry at 5.5 km/s")
    # Returns ProblemSpec with physics_type, materials, boundary conditions, etc.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import os
from dotenv import load_dotenv
from knowledge.rag.vector_db import CortexVectorDB

# Load environment variables
load_dotenv()


@dataclass
class ProblemSpec:
    """
    Specification of an engineering problem extracted by the router.

    This is the structured output that solvers understand.
    """
    # Problem identification
    problem_type: str  # "thermal", "structural", "fluid", "coupled"
    description: str

    # Physics requirements
    physics_types: List[str] = field(default_factory=list)  # ["thermal", "structural"]
    solvers_needed: List[str] = field(default_factory=list)  # ["ThermalSolver", "StructuralSolver"]

    # Geometry
    geometry_type: str = "plate"  # "plate", "cylinder", "custom"
    dimensions: Dict[str, float] = field(default_factory=dict)  # {"width": 0.1, "height": 0.1}

    # Materials
    suggested_materials: List[str] = field(default_factory=list)
    material_requirements: Dict[str, Any] = field(default_factory=dict)

    # Boundary conditions
    boundary_conditions: Dict[str, Dict] = field(default_factory=dict)

    # Loads and constraints
    loads: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

    # Optimization objectives
    objectives: List[str] = field(default_factory=list)  # ["minimize_weight", "maximize_strength"]

    # Confidence
    confidence: float = 0.0  # Router's confidence in the interpretation

    # Raw LLM response for debugging
    raw_response: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "problem_type": self.problem_type,
            "description": self.description,
            "physics_types": self.physics_types,
            "solvers_needed": self.solvers_needed,
            "geometry_type": self.geometry_type,
            "dimensions": self.dimensions,
            "suggested_materials": self.suggested_materials,
            "material_requirements": self.material_requirements,
            "boundary_conditions": self.boundary_conditions,
            "loads": self.loads,
            "constraints": self.constraints,
            "objectives": self.objectives,
            "confidence": self.confidence,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=== Problem Specification ===",
            f"Type: {self.problem_type}",
            f"Description: {self.description}",
            f"Physics: {', '.join(self.physics_types)}",
            f"Solvers: {', '.join(self.solvers_needed)}",
            f"Geometry: {self.geometry_type} {self.dimensions}",
            f"Materials: {', '.join(self.suggested_materials)}",
            f"Confidence: {self.confidence:.0%}",
        ]
        return "\n".join(lines)


class Router:
    """
    Parses natural language engineering problems using Claude API.

    Usage:
        router = Router()
        spec = router.parse("Create a heat sink for 50W LED cooling")
    """

    SYSTEM_PROMPT = """You are an expert computational engineering assistant. Your job is to:
1. Understand engineering problem descriptions
2. Extract specific parameters and requirements
3. Determine what physics simulations are needed
4. Suggest appropriate materials

You must respond in valid JSON format with the following structure:
{
    "problem_type": "thermal" | "structural" | "fluid" | "coupled",
    "description": "Brief technical description",
    "physics_types": ["thermal", "structural"],
    "solvers_needed": ["ThermalSolver"],
    "geometry_type": "plate" | "cylinder" | "box" | "custom",
    "dimensions": {"width": 0.1, "height": 0.1, "thickness": 0.01},
    "suggested_materials": ["aluminum_6061", "copper_c11000"],
    "material_requirements": {
        "min_thermal_conductivity": 100,
        "max_service_temp": 400
    },
    "boundary_conditions": {
        "left": {"type": "temperature", "value": 1200},
        "right": {"type": "convection", "h": 25, "T_inf": 293},
        "top": {"type": "heat_flux", "value": 0},
        "bottom": {"type": "heat_flux", "value": 0}
    },
    "loads": {"heat_load": 50},
    "constraints": {"max_temperature": 358},
    "objectives": ["minimize_temperature", "minimize_weight"],
    "confidence": 0.85
}

Use SI units: meters, Kelvin, Watts, Pascals.
Map internal/external surfaces to boundary names: left, right, top, or bottom.
Example: 'inner surface' -> 'left', 'outer surface' -> 'right'.
If information is missing, make reasonable engineering assumptions and note them.
Available materials: aluminum_6061, stainless_steel_316, inconel_718, titanium_ti6al4v, copper_c11000, silicon_carbide, alumina_al2o3, carbon_carbon, reinforced_carbon_carbon, pica
"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize router with Claude API key and Knowledge Base."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
        self.db = CortexVectorDB()

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
        return self._client

    def parse(self, user_input: str, context: Optional[Dict] = None) -> ProblemSpec:
        """
        Parse natural language input into a ProblemSpec using RAG-enhanced prompts.

        Args:
            user_input: Natural language problem description
            context: Optional additional context (e.g., previous results)

        Returns:
            ProblemSpec with extracted parameters
        """
        # Step 0: RAG Search for engineering knowledge
        print(f"Searching Knowledge Base for: '{user_input}'...")
        rag_results = self.db.search(user_input, limit=3)
        knowledge_context = ""
        if rag_results:
            knowledge_context = "\n--- RELEVANT ENGINEERING KNOWLEDGE ---\n"
            for res in rag_results:
                payload = res.get('payload', {})
                knowledge_context += f"Source: {payload.get('source')}\n"
                knowledge_context += f"Knowledge: {payload.get('content')}\n\n"

        # Build the prompt
        prompt = f"Parse this engineering problem and extract specifications:\n\n{user_input}"
        
        if knowledge_context:
            prompt += f"\n\nUse the following retrieved knowledge to inform your parameters:\n{knowledge_context}"

        if context:
            prompt += f"\n\nAdditional execution context:\n{json.dumps(context, indent=2)}"

        try:
            client = self._get_client()

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = message.content[0].text

            # Parse JSON from response
            spec_dict = self._extract_json(response_text)

            return ProblemSpec(
                problem_type=spec_dict.get("problem_type", "thermal"),
                description=spec_dict.get("description", user_input),
                physics_types=spec_dict.get("physics_types", []),
                solvers_needed=spec_dict.get("solvers_needed", []),
                geometry_type=spec_dict.get("geometry_type", "plate"),
                dimensions=spec_dict.get("dimensions", {}),
                suggested_materials=spec_dict.get("suggested_materials", []),
                material_requirements=spec_dict.get("material_requirements", {}),
                boundary_conditions=spec_dict.get("boundary_conditions", {}),
                loads=spec_dict.get("loads", {}),
                constraints=spec_dict.get("constraints", {}),
                objectives=spec_dict.get("objectives", []),
                confidence=spec_dict.get("confidence", 0.5),
                raw_response=response_text,
            )

        except Exception as e:
            print(f"Warning: LLM parsing failed: {e}")
            print("Falling back to keyword-based parsing...")
            return self._fallback_parse(user_input)

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response text."""
        # Try to find JSON in the response
        import re

        # Look for JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # If no valid JSON found, return empty dict
        return {}

    def _fallback_parse(self, user_input: str) -> ProblemSpec:
        """Fallback keyword-based parsing when LLM is unavailable."""
        input_lower = user_input.lower()

        # Detect problem type
        if any(w in input_lower for w in ["heat", "thermal", "temperature", "cooling", "hot"]):
            problem_type = "thermal"
            physics_types = ["thermal"]
            solvers_needed = ["ThermalSolver"]
        elif any(w in input_lower for w in ["stress", "load", "force", "structural"]):
            problem_type = "structural"
            physics_types = ["structural"]
            solvers_needed = ["StructuralSolver"]
        else:
            problem_type = "thermal"
            physics_types = ["thermal"]
            solvers_needed = ["ThermalSolver"]

        # Detect geometry
        if "plate" in input_lower:
            geometry_type = "plate"
        elif "cylinder" in input_lower:
            geometry_type = "cylinder"
        else:
            geometry_type = "plate"

        # Default dimensions
        dimensions = {"width": 0.1, "height": 0.1, "thickness": 0.01}

        # Suggest materials based on keywords
        suggested_materials = []
        if "aluminum" in input_lower or "lightweight" in input_lower:
            suggested_materials.append("aluminum_6061")
        if "steel" in input_lower:
            suggested_materials.append("stainless_steel_316")
        if "high temp" in input_lower or "heat shield" in input_lower:
            suggested_materials.append("inconel_718")
            suggested_materials.append("carbon_carbon")
        if not suggested_materials:
            suggested_materials = ["aluminum_6061"]

        return ProblemSpec(
            problem_type=problem_type,
            description=user_input,
            physics_types=physics_types,
            solvers_needed=solvers_needed,
            geometry_type=geometry_type,
            dimensions=dimensions,
            suggested_materials=suggested_materials,
            confidence=0.3,  # Low confidence for fallback
        )

    def explain_spec(self, spec: ProblemSpec) -> str:
        """Generate human-readable explanation of the problem spec."""
        lines = [
            "I understood your problem as follows:",
            "",
            f"**Problem Type**: {spec.problem_type.title()} Analysis",
            f"**Description**: {spec.description}",
            "",
            "**What I'll simulate**:",
        ]

        for physics in spec.physics_types:
            lines.append(f"  - {physics.title()} physics")

        lines.append("")
        lines.append("**Geometry**:")
        lines.append(f"  Type: {spec.geometry_type}")
        for dim, val in spec.dimensions.items():
            lines.append(f"  {dim}: {val} m")

        lines.append("")
        lines.append("**Suggested Materials**:")
        for mat in spec.suggested_materials:
            lines.append(f"  - {mat}")

        if spec.boundary_conditions:
            lines.append("")
            lines.append("**Boundary Conditions**:")
            for name, bc in spec.boundary_conditions.items():
                lines.append(f"  - {name}: {bc}")

        lines.append("")
        lines.append(f"**Confidence**: {spec.confidence:.0%}")

        return "\n".join(lines)
