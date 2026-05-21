#!/usr/bin/env python3
"""
Example: Heat Conduction in a Plate

This example demonstrates Cortex CEM's thermal analysis capabilities.

Scenario:
- A 10cm x 10cm aluminum plate
- Left edge heated to 100°C (373K)
- Right edge at room temperature 20°C (293K)
- Top and bottom edges insulated

This is a classic steady-state heat conduction problem.
The solution should show a linear temperature gradient from left to right.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.engine import CortexEngine
from solvers.thermal import ThermalSolver, ThermalProblem, BoundaryCondition
from knowledge import MaterialsDB


def example_1_natural_language():
    """
    Example 1: Natural Language Input

    Let Cortex parse a natural language description and run the analysis.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Natural Language Input")
    print("=" * 60)

    engine = CortexEngine()

    # Natural language problem description
    result = engine.run(
        "Analyze heat conduction in a 10cm x 10cm aluminum plate. "
        "The left edge is heated to 100°C and the right edge is at 20°C. "
        "The top and bottom edges are insulated."
    )

    # Save temperature plot
    try:
        result.plot_temperature(save_path="heat_plate_example1.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    return result


def example_2_direct_api():
    """
    Example 2: Direct API Usage

    Use the solver directly without LLM parsing.
    This gives you full control over the problem setup.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Direct Solver API")
    print("=" * 60)

    # Get material from database
    materials = MaterialsDB()
    aluminum = materials.get("aluminum_6061")

    print(f"\nMaterial: {aluminum.name}")
    print(f"  Thermal conductivity: {aluminum.thermal_conductivity} W/(m·K)")
    print(f"  Density: {aluminum.density} kg/m³")
    print(f"  Specific heat: {aluminum.specific_heat} J/(kg·K)")

    # Define the problem
    problem = ThermalProblem(
        name="Heated Aluminum Plate",
        description="10cm x 10cm plate with temperature gradient",

        # Geometry
        domain_size=(0.1, 0.1),  # 10cm x 10cm
        grid_size=(50, 50),      # 50x50 grid points

        # Material
        material=aluminum,

        # Boundary conditions
        boundary_conditions={
            "left": BoundaryCondition("dirichlet", 373.0),   # 100°C
            "right": BoundaryCondition("dirichlet", 293.0),  # 20°C
            "top": BoundaryCondition("neumann", 0.0),        # Insulated
            "bottom": BoundaryCondition("neumann", 0.0),     # Insulated
        },

        # Solve steady-state
        steady_state=True,
    )

    # Create solver and run
    solver = ThermalSolver(max_iterations=10000, tolerance=1e-6)
    result = solver.solve(problem)

    # Print results
    print(result.summary())

    # Calculate heat transfer rate (Fourier's law)
    # Q = -k * A * dT/dx
    # For steady-state 1D: Q = k * A * (T_hot - T_cold) / L
    k = aluminum.thermal_conductivity
    A = 0.1 * 0.01  # Assuming 1cm thickness
    L = 0.1  # 10cm length
    dT = 373.0 - 293.0  # Temperature difference

    Q = k * A * dT / L
    print(f"\nCalculated heat transfer rate: {Q:.2f} W")

    return result


def example_3_convection_bc():
    """
    Example 3: Convection Boundary Condition

    Heat sink cooling with convection on one side.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Convection Cooling")
    print("=" * 60)

    materials = MaterialsDB()
    copper = materials.get("copper_c11000")

    print(f"\nMaterial: {copper.name}")
    print(f"  Thermal conductivity: {copper.thermal_conductivity} W/(m·K)")

    # Heat sink with heat source on bottom, convection on top
    problem = ThermalProblem(
        name="Heat Sink with Convection",
        description="Copper heat sink with air cooling",

        domain_size=(0.05, 0.02),  # 5cm x 2cm
        grid_size=(50, 20),

        material=copper,

        boundary_conditions={
            # Bottom: heat source (applied as fixed high temperature)
            "bottom": BoundaryCondition("dirichlet", 353.0),  # 80°C

            # Top: convection to air
            "top": BoundaryCondition("convection", 293.0, coefficient=25.0),  # h=25 W/(m²·K), T_air=20°C

            # Sides: insulated (symmetry)
            "left": BoundaryCondition("neumann", 0.0),
            "right": BoundaryCondition("neumann", 0.0),
        },

        steady_state=True,
    )

    solver = ThermalSolver()
    result = solver.solve(problem)

    print(result.summary())

    # The temperature at the top surface should be between 80°C and 20°C
    print(f"\nTop surface temperature: ~{result.temperature[-1, result.temperature.shape[1]//2]:.1f} K")

    return result


def example_4_transient():
    """
    Example 4: Transient Heat Conduction

    Watch temperature evolve over time as heat spreads through the material.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Transient Analysis")
    print("=" * 60)

    materials = MaterialsDB()
    steel = materials.get("stainless_steel_316")

    print(f"\nMaterial: {steel.name}")
    print(f"  Thermal diffusivity: {steel.thermal_diffusivity():.2e} m²/s")

    # Initially uniform temperature, then left edge suddenly heated
    problem = ThermalProblem(
        name="Transient Heat Conduction",
        description="Steel plate with sudden heating on left edge",

        domain_size=(0.1, 0.1),
        grid_size=(40, 40),

        material=steel,

        # Initially at room temperature
        initial_temperature=293.0,  # 20°C

        boundary_conditions={
            "left": BoundaryCondition("dirichlet", 373.0),   # Suddenly 100°C
            "right": BoundaryCondition("neumann", 0.0),       # Insulated
            "top": BoundaryCondition("neumann", 0.0),
            "bottom": BoundaryCondition("neumann", 0.0),
        },

        # Transient analysis
        steady_state=False,
        total_time=60.0,     # 60 seconds
        time_steps=1000,
    )

    solver = ThermalSolver()
    result = solver.solve(problem)

    print(result.summary())
    print(f"\nAfter {problem.total_time}s:")
    print(f"  Temperature has propagated {(result.T_max - 293) / (373 - 293) * 100:.1f}% into the plate")

    return result


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("     CORTEX CEM - Heat Conduction Examples")
    print("=" * 60)

    # Example 1: Natural language (requires Claude API)
    try:
        example_1_natural_language()
    except Exception as e:
        print(f"\nExample 1 skipped: {e}")
        print("(This example requires ANTHROPIC_API_KEY in .env)")

    # Example 2: Direct API
    example_2_direct_api()

    # Example 3: Convection
    example_3_convection_bc()

    # Example 4: Transient
    example_4_transient()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
