"""
Pipeline Executor for Cortex CEM

Orchestrates the execution of multiple solvers in the correct order,
passing results between them.

Example pipeline:
    1. Parse problem → ProblemSpec
    2. Select materials → Material
    3. Run thermal solver → Temperature field
    4. Run structural solver (with thermal loads) → Stress field
    5. Validate results → Pass/Fail
    6. Generate geometry → SDF/Mesh
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import time

from .router import ProblemSpec
from knowledge import MaterialsDB, PhysicsRules, DesignPatterns
from solvers import ThermalSolver, ThermalProblem
from solvers.thermal import BoundaryCondition
from geometry.shapes.Primitives import Box, Cylinder


class StepStatus(Enum):
    """Status of a pipeline step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """A single step in the execution pipeline."""
    name: str
    description: str
    solver_type: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class Pipeline:
    """
    A pipeline of solver steps to execute.

    The pipeline is a directed acyclic graph (DAG) of steps,
    where each step can depend on outputs from previous steps.
    """
    name: str
    steps: List[PipelineStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: PipelineStep):
        """Add a step to the pipeline."""
        self.steps.append(step)

    def get_step(self, name: str) -> Optional[PipelineStep]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def summary(self) -> str:
        """Get pipeline execution summary."""
        lines = [f"=== Pipeline: {self.name} ===", ""]

        for i, step in enumerate(self.steps, 1):
            status_icon = {
                StepStatus.PENDING: "⏳",
                StepStatus.RUNNING: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️",
            }.get(step.status, "?")

            lines.append(f"{i}. {status_icon} {step.name} ({step.status.value})")
            if step.execution_time > 0:
                lines.append(f"   Time: {step.execution_time:.3f}s")
            if step.error:
                lines.append(f"   Error: {step.error}")

        return "\n".join(lines)


class PipelineExecutor:
    """
    Executes computational engineering pipelines.

    Takes a ProblemSpec and:
    1. Builds the appropriate pipeline of solvers
    2. Executes each step in order
    3. Passes results between steps
    4. Validates final results

    Usage:
        executor = PipelineExecutor()
        pipeline = executor.build_pipeline(problem_spec)
        results = executor.run(pipeline)
    """

    def __init__(self):
        self.materials_db = MaterialsDB()
        self.physics_rules = PhysicsRules()
        self.design_patterns = DesignPatterns()

        # Available solvers
        self.solvers = {
            "ThermalSolver": ThermalSolver(),
        }

    def build_pipeline(self, spec: ProblemSpec) -> Pipeline:
        """Build a pipeline from a problem specification."""
        pipeline = Pipeline(name=f"Pipeline for: {spec.description[:50]}...")

        # Step 1: Material selection
        pipeline.add_step(PipelineStep(
            name="material_selection",
            description="Select appropriate materials based on requirements",
        ))

        # Step 2: Add solver steps based on physics types
        for physics_type in spec.physics_types:
            if physics_type == "thermal":
                pipeline.add_step(PipelineStep(
                    name="thermal_analysis",
                    description="Solve heat conduction/transfer",
                    solver_type="ThermalSolver",
                    depends_on=["material_selection"],
                ))
            elif physics_type == "structural":
                # Structural analysis may depend on thermal results
                depends = ["material_selection"]
                if "thermal" in spec.physics_types:
                    depends.append("thermal_analysis")

                pipeline.add_step(PipelineStep(
                    name="structural_analysis",
                    description="Solve stress/strain",
                    solver_type="StructuralSolver",
                    depends_on=depends,
                ))

        pipeline.add_step(PipelineStep(
            name="validation",
            description="Validate results against constraints",
            depends_on=[s.name for s in pipeline.steps if "analysis" in s.name],
        ))

        # Step 4: Geometry Generation
        pipeline.add_step(PipelineStep(
            name="geometry_generation",
            description="Generate 3D geometry from problem spec",
            depends_on=["validation"],
        ))

        # Store spec in context
        pipeline.context["spec"] = spec

        return pipeline

    def run(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Execute the pipeline."""
        results = {}

        for step in pipeline.steps:
            # Check dependencies
            deps_satisfied = all(
                pipeline.get_step(dep).status == StepStatus.COMPLETED
                for dep in step.depends_on
            )

            if not deps_satisfied:
                step.status = StepStatus.SKIPPED
                step.error = "Dependencies not satisfied"
                continue

            step.status = StepStatus.RUNNING
            start_time = time.time()

            try:
                if step.name == "material_selection":
                    step.result = self._select_material(pipeline.context["spec"])
                    pipeline.context["material"] = step.result

                elif step.name == "thermal_analysis":
                    step.result = self._run_thermal(
                        pipeline.context["spec"],
                        pipeline.context.get("material")
                    )
                    pipeline.context["thermal_result"] = step.result

                elif step.name == "structural_analysis":
                    step.result = self._run_structural(
                        pipeline.context["spec"],
                        pipeline.context.get("material"),
                        pipeline.context.get("thermal_result")
                    )
                    pipeline.context["structural_result"] = step.result

                elif step.name == "validation":
                    step.result = self._validate_results(pipeline.context)

                elif step.name == "geometry_generation":
                    step.result = self._generate_geometry(pipeline.context)
                    pipeline.context["geometry"] = step.result

                step.status = StepStatus.COMPLETED
                results[step.name] = step.result

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                results[step.name] = {"error": str(e)}

            step.execution_time = time.time() - start_time

        results["pipeline_summary"] = pipeline.summary()
        results["context"] = pipeline.context

        return results

    def _select_material(self, spec: ProblemSpec):
        """Select the best material based on requirements."""
        # Try suggested materials first
        for mat_name in spec.suggested_materials:
            material = self.materials_db.get(mat_name)
            if material:
                return material

        # Fallback: find materials matching requirements
        requirements = spec.material_requirements

        if "min_thermal_conductivity" in requirements:
            candidates = self.materials_db.find_by_conductivity(
                min_k=requirements["min_thermal_conductivity"]
            )
            if candidates:
                return candidates[0]

        if "max_service_temp" in requirements:
            candidates = self.materials_db.find_by_max_temp(
                min_temp=requirements["max_service_temp"]
            )
            if candidates:
                return candidates[0]

        # Default to aluminum
        return self.materials_db.get("aluminum_6061")

    def _run_thermal(self, spec: ProblemSpec, material):
        """Run thermal analysis."""
        solver = self.solvers.get("ThermalSolver")
        if not solver:
            raise RuntimeError("ThermalSolver not available")

        # Build thermal problem from spec
        dims = spec.dimensions
        width = dims.get("width", 0.1)
        height = dims.get("height", 0.1)

        # Parse boundary conditions from spec
        bcs = {}
        for name, bc_spec in spec.boundary_conditions.items():
            if isinstance(bc_spec, dict):
                bc_type = bc_spec.get("type", "neumann")
                value = bc_spec.get("value", 0.0)

                if bc_type == "heat_flux":
                    bcs[name] = BoundaryCondition("neumann", value)
                elif bc_type == "temperature":
                    bcs[name] = BoundaryCondition("dirichlet", value)
                elif bc_type == "convection":
                    h = bc_spec.get("h", 25.0)
                    T_inf = bc_spec.get("T_inf", 293.0)
                    bcs[name] = BoundaryCondition("convection", T_inf, h)

        # Default BCs if not specified
        if not bcs:
            bcs = {
                "left": BoundaryCondition("dirichlet", 373.0),   # Hot side
                "right": BoundaryCondition("dirichlet", 293.0),  # Cold side
                "top": BoundaryCondition("neumann", 0.0),        # Insulated
                "bottom": BoundaryCondition("neumann", 0.0),     # Insulated
            }

        problem = ThermalProblem(
            name=spec.description[:50],
            domain_size=(width, height),
            grid_size=(50, 50),
            material=material,
            boundary_conditions=bcs,
            steady_state=True,
        )

        return solver.solve(problem)

    def _run_structural(self, spec: ProblemSpec, material, thermal_result=None):
        """Run structural analysis."""
        # Placeholder - StructuralSolver not yet implemented
        return {
            "status": "skipped",
            "message": "Structural solver not yet implemented",
            "thermal_input": thermal_result is not None,
        }

    def _validate_results(self, context: Dict) -> Dict:
        """Validate results against constraints."""
        spec = context.get("spec")
        results = {"valid": True, "checks": [], "warnings": []}

        # Check thermal results
        thermal_result = context.get("thermal_result")
        if thermal_result:
            T_max = thermal_result.T_max
            T_min = thermal_result.T_min

            # Check against constraints
            constraints = spec.constraints if spec else {}

            if "max_temperature" in constraints:
                limit = constraints["max_temperature"]
                if T_max > limit:
                    results["valid"] = False
                    results["checks"].append(
                        f"FAIL: Max temperature {T_max:.1f}K exceeds limit {limit}K"
                    )
                else:
                    results["checks"].append(
                        f"PASS: Max temperature {T_max:.1f}K within limit {limit}K"
                    )

            # Material temperature check
            material = context.get("material")
            if material and T_max > material.max_service_temp:
                results["valid"] = False
                results["checks"].append(
                    f"FAIL: Temperature {T_max:.1f}K exceeds material limit {material.max_service_temp}K"
                )
            elif material:
                results["checks"].append(
                    f"PASS: Temperature {T_max:.1f}K within material limit {material.max_service_temp}K"
                )

        return results

    def _generate_geometry(self, context: Dict) -> Any:
        """Generate geometry based on problem spec."""
        spec = context.get("spec")
        if not spec:
            return None
            
        print(f"Generating 3D {spec.geometry_type} geometry...")
        dims = spec.dimensions
        
        if spec.geometry_type == "plate" or spec.geometry_type == "box":
            w = dims.get("width", 0.1)
            h = dims.get("height", 0.1)
            t = dims.get("thickness", 0.01)
            shape = Box(size=[w, h, t])
            return shape
        elif spec.geometry_type == "cylinder":
            r = dims.get("radius", 0.05)
            h = dims.get("height", 0.1)
            shape = Cylinder(radius=r, height=h)
            return shape
            
        return None
