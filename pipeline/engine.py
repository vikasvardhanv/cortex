"""
Cortex Engine - Main entry point for the CEM system.

This is the high-level interface that users interact with.
It combines the Router, Pipeline Executor, and Geometry Engine.

Usage:
    from cortex import CortexEngine

    engine = CortexEngine()
    result = engine.run("Create a heat sink for 50W LED cooling")
    print(result.summary())
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
import mlflow
import os

from .router import Router, ProblemSpec
from .executor import PipelineExecutor, Pipeline
from knowledge import MaterialsDB, PhysicsRules


@dataclass
class CortexResult:
    """
    Complete result from Cortex CEM analysis.

    Contains:
    - Problem specification (what was understood)
    - Solver results (temperature, stress, etc.)
    - Validation status
    - Geometry output (if generated)
    """
    problem_spec: ProblemSpec
    pipeline_results: Dict[str, Any]
    validation: Dict[str, Any]
    geometry: Optional[Any] = None  # Will hold SDF/Mesh

    # Summary statistics
    success: bool = False
    execution_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "           CORTEX CEM ANALYSIS RESULT",
            "=" * 60,
            "",
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Execution Time: {self.execution_time:.2f}s",
            "",
            "--- Problem ---",
            f"Type: {self.problem_spec.problem_type}",
            f"Description: {self.problem_spec.description}",
            "",
        ]

        # Material info
        material = self.pipeline_results.get("context", {}).get("material")
        if material:
            lines.extend([
                "--- Material ---",
                f"Name: {material.name}",
                f"Thermal Conductivity: {material.thermal_conductivity} W/(m·K)",
                f"Max Service Temp: {material.max_service_temp} K",
                "",
            ])

        # Thermal results
        thermal = self.pipeline_results.get("context", {}).get("thermal_result")
        if thermal:
            lines.extend([
                "--- Thermal Analysis ---",
                f"Temperature Range: {thermal.T_min:.1f}K to {thermal.T_max:.1f}K",
                f"                  ({thermal.T_min - 273.15:.1f}°C to {thermal.T_max - 273.15:.1f}°C)",
                f"Average Temperature: {thermal.T_avg:.1f}K ({thermal.T_avg - 273.15:.1f}°C)",
                f"Solver Iterations: {thermal.iterations}",
                f"Converged: {thermal.converged}",
                "",
            ])

        # Validation
        if self.validation:
            lines.extend([
                "--- Validation ---",
                f"Valid: {self.validation.get('valid', 'N/A')}",
            ])
            for check in self.validation.get("checks", []):
                lines.append(f"  {check}")
            lines.append("")

        # Geometry info
        if self.geometry:
            lines.extend([
                "--- Geometry ---",
                f"Type: {self.problem_spec.geometry_type}",
                f"Generated: {type(self.geometry).__name__}",
                f"Dimensions: {self.problem_spec.dimensions}",
                "",
            ])

        # Warnings and errors
        if self.warnings:
            lines.append("--- Warnings ---")
            for w in self.warnings:
                lines.append(f"  ⚠️  {w}")
            lines.append("")

        if self.errors:
            lines.append("--- Errors ---")
            for e in self.errors:
                lines.append(f"  ❌ {e}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def get_temperature_field(self) -> Optional[np.ndarray]:
        """Get temperature field from thermal analysis."""
        thermal = self.pipeline_results.get("context", {}).get("thermal_result")
        if thermal:
            return thermal.temperature
        return None

    def plot_temperature(self, save_path: Optional[str] = None):
        """Plot temperature distribution."""
        import matplotlib.pyplot as plt

        thermal = self.pipeline_results.get("context", {}).get("thermal_result")
        if thermal is None or thermal.temperature is None:
            print("No temperature data available")
            return

        T = thermal.temperature
        x = thermal.x_coords
        y = thermal.y_coords

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create meshgrid for plotting
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Plot temperature contours
        contour = ax.contourf(X * 1000, Y * 1000, T, levels=50, cmap='hot')
        cbar = plt.colorbar(contour, ax=ax, label='Temperature (K)')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title(f'Temperature Distribution\n{self.problem_spec.description[:50]}')
        ax.set_aspect('equal')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

        return fig


class CortexEngine:
    """
    Main Cortex CEM Engine.

    This is the primary interface for running computational engineering analyses.
    It handles:
    1. Natural language parsing (Router)
    2. Solver orchestration (Pipeline Executor)
    3. Result validation
    4. Geometry generation

    Usage:
        engine = CortexEngine()

        # Natural language input
        result = engine.run("Design a heat sink plate for 100W heat load")

        # Or structured input
        result = engine.analyze(
            problem_type="thermal",
            geometry={"type": "plate", "width": 0.1, "height": 0.1},
            material="aluminum_6061",
            boundary_conditions={...}
        )
    """

    def __init__(self, api_key: Optional[str] = None, verbose: bool = True):
        """
        Initialize Cortex Engine.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            verbose: Print progress messages
        """
        self.router = Router(api_key=api_key)
        self.executor = PipelineExecutor()
        self.verbose = verbose

        # Knowledge bases (accessible for direct queries)
        self.materials = MaterialsDB()
        self.physics = PhysicsRules()

    def run(self, user_input: str) -> CortexResult:
        """
        Run analysis from natural language input.

        Args:
            user_input: Natural language description of the engineering problem

        Returns:
            CortexResult with complete analysis results
        """
        import time
        start_time = time.time()

        if self.verbose:
            print("=" * 60)
            print("CORTEX CEM - Computational Engineering Model")
            print("=" * 60)
            print(f"\nInput: {user_input}\n")

        # Step 1: Parse input
        if self.verbose:
            print("Step 1: Parsing problem specification...")

        spec = self.router.parse(user_input)

        if self.verbose:
            print(f"  Problem type: {spec.problem_type}")
            print(f"  Physics: {spec.physics_types}")
            print(f"  Materials: {spec.suggested_materials}")
            print(f"  Confidence: {spec.confidence:.0%}")
            print()

        # Step 2: Build and execute pipeline
        if self.verbose:
            print("Step 2: Building execution pipeline...")

        pipeline = self.executor.build_pipeline(spec)

        if self.verbose:
            print(f"  Steps: {[s.name for s in pipeline.steps]}")
            print()
            print("Step 3: Executing pipeline...")

        results = self.executor.run(pipeline)

        if self.verbose:
            print(pipeline.summary())
            print()

        # Step 3: Extract validation
        validation = results.get("validation", {"valid": True, "checks": []})

        # Build result object
        result = CortexResult(
            problem_spec=spec,
            pipeline_results=results,
            validation=validation,
            geometry=results.get("geometry_generation"),
            success=validation.get("valid", False),
            execution_time=time.time() - start_time,
        )

        # Collect warnings and errors
        for step in pipeline.steps:
            if step.error:
                result.errors.append(f"{step.name}: {step.error}")

        if spec.confidence < 0.5:
            result.warnings.append(
                f"Low parsing confidence ({spec.confidence:.0%}) - results may not match intent"
            )

        if self.verbose:
            print(result.summary())

        # Final Step: Log to MLflow
        try:
            mlflow.set_experiment("Cortex-Inference")
            with mlflow.start_run(run_name=f"Run_{int(time.time())}"):
                mlflow.log_params({
                    "problem_type": spec.problem_type,
                    "input_description": user_input[:250],
                    "material_suggested": spec.suggested_materials[0] if spec.suggested_materials else "generic"
                })
                
                # Log thermal metrics if available
                thermal = result.pipeline_results.get("context", {}).get("thermal_result")
                if thermal:
                    mlflow.log_metric("T_max", thermal.T_max)
                    mlflow.log_metric("T_min", thermal.T_min)
                    mlflow.log_metric("converged", 1 if thermal.converged else 0)
                
                mlflow.log_metric("success", 1 if result.success else 0)
                mlflow.log_metric("execution_time", result.execution_time)
                
                # Create and log plots if thermal
                if thermal:
                    plot_path = f"results/plot_{int(time.time())}.png"
                    os.makedirs("results", exist_ok=True)
                    result.plot_temperature(save_path=plot_path)
                    mlflow.log_artifact(plot_path)
        except Exception as e:
            if self.verbose:
                print(f"MLflow Logging skipped: {e}")

        return result

    def analyze(
        self,
        problem_type: str,
        geometry: Dict[str, Any],
        material: str,
        boundary_conditions: Dict[str, Dict],
        **kwargs
    ) -> CortexResult:
        """
        Run analysis with structured input (no LLM parsing).

        Args:
            problem_type: "thermal", "structural", or "coupled"
            geometry: {"type": "plate", "width": 0.1, "height": 0.1}
            material: Material name from database
            boundary_conditions: Dict of boundary conditions
            **kwargs: Additional parameters

        Returns:
            CortexResult with analysis results
        """
        # Build ProblemSpec directly
        spec = ProblemSpec(
            problem_type=problem_type,
            description=f"Direct {problem_type} analysis",
            physics_types=[problem_type],
            solvers_needed=[f"{problem_type.title()}Solver"],
            geometry_type=geometry.get("type", "plate"),
            dimensions={k: v for k, v in geometry.items() if k != "type"},
            suggested_materials=[material],
            boundary_conditions=boundary_conditions,
            constraints=kwargs.get("constraints", {}),
            confidence=1.0,  # Full confidence for structured input
        )

        # Execute
        pipeline = self.executor.build_pipeline(spec)
        results = self.executor.run(pipeline)
        validation = results.get("validation", {"valid": True, "checks": []})

        return CortexResult(
            problem_spec=spec,
            pipeline_results=results,
            validation=validation,
            success=validation.get("valid", False),
        )

    def list_materials(self) -> List[str]:
        """List available materials."""
        return self.materials.list_all()

    def get_material_info(self, name: str) -> Optional[Dict]:
        """Get detailed material information."""
        mat = self.materials.get(name)
        if mat:
            return mat.to_dict()
        return None

    def help(self) -> str:
        """Show help information."""
        return """
CORTEX CEM - Computational Engineering Model
============================================

Cortex is a computational engineering system that can:
- Parse natural language engineering problems
- Run physics simulations (thermal, structural, fluid)
- Validate designs against constraints
- Generate optimized geometry

QUICK START:
------------
from cortex import CortexEngine

engine = CortexEngine()
result = engine.run("Design a heat sink for 50W LED cooling")
print(result.summary())
result.plot_temperature()

SUPPORTED ANALYSES:
-------------------
- Thermal: Heat conduction, convection, radiation
- Structural: Stress, strain, deformation (coming soon)
- Fluid: CFD simulation (coming soon)
- Coupled: Multi-physics (coming soon)

MATERIALS DATABASE:
-------------------
Use engine.list_materials() to see available materials.
Use engine.get_material_info("aluminum_6061") for details.

EXAMPLE PROMPTS:
----------------
- "Analyze heat flow in a 10cm x 10cm aluminum plate with left edge at 100°C"
- "Design a heat shield for atmospheric re-entry"
- "Create a bracket to support 1000N load"
"""
