
# Lattice Structures (Gyroids/TPMS)
from geometry.core import SDF, BBox3
import numpy as np

class Gyroid(SDF):
    """Triply Periodic Minimal Surface (Gyroid)"""
    def __init__(self, period=10.0, thickness=1.0, center=[0,0,0]):
        self.period = period
        self.thickness = thickness
        self.center = np.array(center)
        self.scale = 2 * np.pi / period
    
    def dist(self, p):
        # p is (N, 3)
        # Gyroid equation: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
        q = (p - self.center) * self.scale
        val = (np.sin(q[:,0]) * np.cos(q[:,1]) + 
               np.sin(q[:,1]) * np.cos(q[:,2]) + 
               np.sin(q[:,2]) * np.cos(q[:,0]))
        
        # Approximate distance field based on gradient
        # Typically abs(val) / gradient_magnitude, but gradient varies.
        # For small thickness, val / 1.5 is a decent approximation.
        return np.abs(val) / 1.5 - (self.thickness * self.scale / 2.0)
    
    def bounds(self):
        # Infinite structure, so returns huge bounds or user must limit it
        # Here we return a default large box
        return BBox3([-1000,-1000,-1000], [1000,1000,1000])

