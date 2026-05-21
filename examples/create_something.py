#!/usr/bin/env python3
"""
Cortex CEM - Creation Example
This script demonstrates how to create a validated engineering component from natural language.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.engine import CortexEngine

def create_component():
    # 1. Initialize the Engine
    # This loads the Router (LLM + RAG), Solvers, and Geometry Engine
    print("Initializing Cortex Engine...")
    engine = CortexEngine()

    # 2. Define your engineering task in natural language
    # Note how we specify a material (Inconel 718) that we just added to our RAG knowledge base.
    prompt = (
        "Design a thermal protection plate for a rocket engine nozzle extension. "
        "The plate is 20cm x 20cm and made of Inconel 718. "
        "The inner surface is exposed to 1200K combustion gases, "
        "while the outer surface is cooled by radiation to space (3K)."
    )

    print("\n" + "="*60)
    print("CREATION TASK")
    print("="*60)
    print(f"Goal: {prompt}")

    # 3. Run the Cortex Pipeline
    # This:
    # - Searches the RAG for Inconel 718 properties and radiation equations
    # - Parses the prompt into a physics problem spec
    # - Executes the Thermal Solver
    # - Validates if Inconel 718 can survive 1200K
    result = engine.run(prompt)

    # 4. Interact with the Results
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(result.summary())

    # 5. Visualizing the 'Creation'
    # We can plot the temperature field that was 'created' by the solver
    print("\nPlotting temperature distribution...")
    result.plot_temperature(save_path="nozzle_extension_thermal.png")
    print("Thermal map saved to 'nozzle_extension_thermal.png'")

    # 6. Accessing Structured Data and Exporting Geometry
    temp_field = result.get_temperature_field()
    if temp_field is not None:
        max_temp = temp_field.max()
        print(f"\nFinal Calculated Peak Temperature: {max_temp:.2f} K")
        
        # Check against material limits retrieved from RAG
        material = result.pipeline_results.get("context", {}).get("material")
        if material:
            mat_limit = material.max_service_temp
            if max_temp < mat_limit:
                print(f"✅ Design Validated: {material.name} limit is {mat_limit}K. We are safe.")
            else:
                print(f"❌ Design Warning: Temperature exceeds {material.name} limit of {mat_limit}K!")

        # 7. OBJ Export (New!)
        if result.geometry:
            print("\nMeshing 3D geometry for export...")
            from geometry.core.Voxel import VoxelGrid
            # Convert SDF to Voxel Grid, then Mesh
            grid = VoxelGrid(resolution=0.002) # 2mm resolution
            grid.from_sdf(result.geometry)
            mesh = grid.meshing()
            
            output_file = "created_heat_shield.obj"
            mesh.save_obj(output_file)
            print(f"🚀 3D Model saved to '{output_file}'")

if __name__ == "__main__":
    create_component()
