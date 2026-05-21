
# Implementation of a Rocket Nozzle using Cortex Geometry Engine
# This script demonstrates the use of SDFs to procedurally generate a rocket nozzle

from geometry import (
    SDF, Union, Difference, Intersection, Offset, SmoothUnion, VoxelGrid,
    Sphere, Cylinder, Box, Torus
)
import numpy as np
import time

# --- 1. Define the Nozzle Geometry ---
# A Rocket Nozzle is essentially a revolved profile.
# We will construct it using constructive solid geometry (CSG).

class Cone(SDF):
    """Simple Cone SDF for throat/bell"""
    def __init__(self, angle=np.radians(30), height=10.0, center=[0,0,0]):
        self.angle = angle
        self.height = height
        self.center = np.array(center)
        self.tan_a = np.tan(angle)
        self.cos_a = np.cos(angle)

    def dist(self, p):
        # p is (N, 3)
        # q = length(p.xy)
        p = p - self.center
        q = np.linalg.norm(p[:, :2], axis=1)
        # SDF of a cone
        # float q = length(p.xy);
        # return dot(c*q,s*p.z)
        
        # Simplified: distance to the cone surface
        # r = z * tan(angle)
        # d = (q - z * tan_a) * cos_a
        return (q - p[:, 2] * self.tan_a) * self.cos_a

    def bounds(self):
        # Rough bounds
        r = self.height * self.tan_a
        return Box([r*2, r*2, self.height], self.center + [0,0,self.height/2]).bounds()

# Define the components
print("Defining Geometry...")

# 1. Main Chamber (Cylinder)
chamber_r = 15.0 # mm
chamber_h = 40.0 # mm
chamber = Cylinder(radius=chamber_r, height=chamber_h, center=[0, 0, chamber_h/2])

# 2. Throat (Torus / Smooth Union)
throat_r = 8.0 # mm
throat_pos = [0, 0, chamber_h] 
# Approximate throat transition with smooth union to nozzle
# Or a specific torus shape.
# Let's use a simple approach: Chamber + Bell Cone combined smoothly.

# 3. Bell Nozzle (Cone)
bell_h = 60.0
bell_end_r = 25.0
# Calculate angle for cone
bell_angle = np.arctan2(bell_end_r - throat_r, bell_h)
bell = Cone(angle=bell_angle, height=bell_h, center=[0, 0, chamber_h])

# Combine Chamber and Bell
# We want a smooth transition at the throat
outer_shape = Union(chamber, bell)

# Apply some smoothing at the intersection (z=chamber_h)
# In MyCEM, Union is sharp. SmoothUnion is expensive but nice.
# For now, let's keep it sharp or use a small smoothing factor.
smooth_shape = SmoothUnion(chamber, bell, k=5.0)

# Create the internal void (The actual flow path)
# The wall thickness should be constant or variable.
# We can achieve this by offsetting the outer shape inwards.
wall_thickness = 2.0 # mm
inner_shape = Offset(smooth_shape, -wall_thickness)

# Final Solid: Outer - Inner
rocket_nozzle = Difference(smooth_shape, inner_shape)

# Add Flange at the bottom
flange = Cylinder(radius=chamber_r + 5.0, height=5.0, center=[0, 0, 0])
rocket_nozzle = Union(rocket_nozzle, flange)

# Cut off the top and bottom to ensure open ends if needed
# The cone goes on forever in SDF math, so we need to intersect with a bounding box
# or subtract a box from the top.
cutoff_box = Box(size=[100, 100, chamber_h + bell_h], center=[0, 0, (chamber_h + bell_h)/2])
rocket_nozzle = Intersection(rocket_nozzle, cutoff_box)


# --- 2. Voxelization ---
print("Voxelizing (Resolution: 0.5mm)...")
t0 = time.time()
grid = VoxelGrid(resolution=0.5)
# Reduce resolution for speed in demo (0.5mm is coarse but fast)
# Real engineering might use 0.1mm

grid.from_sdf(rocket_nozzle, padding=2.0)
print(f"Voxelization took {time.time()-t0:.2f}s")
print(f"Grid Size: {grid.size}")

# --- 3. Meshing & Export ---
print("Generating Mesh...")
t0 = time.time()
mesh = grid.meshing()
print(f"Meshing took {time.time()-t0:.2f}s")

output_file = "rocket_nozzle.obj"
mesh.save_obj(output_file)
print(f"Done! Saved to {output_file}")
