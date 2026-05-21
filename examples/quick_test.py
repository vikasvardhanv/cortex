#!/usr/bin/env python3
"""
Quick test script for Cortex CEM.

Run this to verify the installation works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")

    from knowledge import MaterialsDB, PhysicsRules, DesignPatterns
    from solvers import ThermalSolver, ThermalProblem
    from solvers.thermal import BoundaryCondition
    from pipeline import Router, PipelineExecutor, CortexEngine

    print("  ✓ All imports successful")
    return True


def test_materials_db():
    """Test materials database."""
    print("\nTesting MaterialsDB...")

    from knowledge import MaterialsDB

    db = MaterialsDB()
    materials = db.list_all()
    print(f"  Found {len(materials)} materials")

    aluminum = db.get("aluminum_6061")
    assert aluminum is not None, "Failed to get aluminum"
    print(f"  ✓ Aluminum 6061: k={aluminum.thermal_conductivity} W/(m·K)")

    high_temp = db.find_by_max_temp(1500)
    print(f"  ✓ Found {len(high_temp)} materials for T > 1500K")

    return True


def test_physics_rules():
    """Test physics rules."""
    print("\nTesting PhysicsRules...")

    from knowledge import PhysicsRules
    from knowledge.physics_rules import PhysicsType

    rules = PhysicsRules()

    # Test classification
    problem_type = rules.classify_problem(["heat", "temperature", "plate"])
    assert problem_type == PhysicsType.THERMAL
    print(f"  ✓ Classified 'heat, temperature, plate' as {problem_type.value}")

    # Test calculation
    q = rules.heat_flux_conduction(k=200, dT=80, dx=0.1)
    print(f"  ✓ Heat flux calculation: q = {q:.0f} W/m²")

    return True


def test_thermal_solver():
    """Test thermal solver with a simple problem."""
    print("\nTesting ThermalSolver...")

    from solvers import ThermalSolver, ThermalProblem
    from solvers.thermal import BoundaryCondition
    from knowledge import MaterialsDB

    db = MaterialsDB()
    aluminum = db.get("aluminum_6061")

    problem = ThermalProblem(
        name="Test Problem",
        domain_size=(0.1, 0.1),
        grid_size=(20, 20),
        material=aluminum,
        boundary_conditions={
            "left": BoundaryCondition("dirichlet", 373.0),
            "right": BoundaryCondition("dirichlet", 293.0),
            "top": BoundaryCondition("neumann", 0.0),
            "bottom": BoundaryCondition("neumann", 0.0),
        },
        steady_state=True,
    )

    solver = ThermalSolver()
    result = solver.solve(problem)

    print(f"  ✓ Solved in {result.solve_time:.3f}s, {result.iterations} iterations")
    print(f"  ✓ Temperature range: {result.T_min:.1f}K to {result.T_max:.1f}K")
    print(f"  ✓ Converged: {result.converged}")

    # Verify the solution makes sense
    assert result.T_min >= 293.0 - 1  # Should be close to cold boundary
    assert result.T_max <= 373.0 + 1  # Should be close to hot boundary
    assert result.converged

    return True


def test_router_fallback():
    """Test router with fallback (no API key needed)."""
    print("\nTesting Router (fallback mode)...")

    from pipeline import Router

    router = Router(api_key="fake_key")  # Force fallback

    # Test fallback parsing
    spec = router._fallback_parse("Create a heat sink for LED cooling")

    assert spec.problem_type == "thermal"
    assert "thermal" in spec.physics_types
    print(f"  ✓ Parsed problem type: {spec.problem_type}")
    print(f"  ✓ Suggested materials: {spec.suggested_materials}")

    return True


def test_pipeline():
    """Test pipeline execution."""
    print("\nTesting Pipeline Executor...")

    from pipeline import PipelineExecutor, Router

    router = Router(api_key="fake_key")
    spec = router._fallback_parse("Heat a 10cm aluminum plate")

    executor = PipelineExecutor()
    pipeline = executor.build_pipeline(spec)

    print(f"  ✓ Built pipeline with {len(pipeline.steps)} steps")
    print(f"  ✓ Steps: {[s.name for s in pipeline.steps]}")

    results = executor.run(pipeline)
    print(f"  ✓ Pipeline executed successfully")

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("   CORTEX CEM - Quick Test Suite")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Materials DB", test_materials_db),
        ("Physics Rules", test_physics_rules),
        ("Thermal Solver", test_thermal_solver),
        ("Router Fallback", test_router_fallback),
        ("Pipeline", test_pipeline),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\n✓ All tests passed! Cortex CEM is ready to use.")
        print("\nTry running:")
        print("  python examples/heat_conduction_plate.py")
    else:
        print("\n✗ Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
