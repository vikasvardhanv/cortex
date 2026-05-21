
from knowledge.rag.vector_db import CortexVectorDB
import json

class AerospaceMaterialsIngestor:
    def __init__(self):
        self.db = CortexVectorDB()

    def run(self):
        materials = [
            {
                "name": "Aluminum 6061-T6",
                "properties": {
                    "thermal_conductivity": 167.0,
                    "specific_heat": 896.0,
                    "melting_range": "582-652 °C",
                    "density": 2700.0,
                    "thermal_expansion": 23.6e-6,
                    "youngs_modulus": 68.9e9,
                    "yield_strength": 276e6,
                    "max_service_temp": "300 °C"
                },
                "tags": ["aerospace", "aluminum", "common", "structural"],
                "source": "NASA Materials Handbooks / ASM International"
            },
            {
                "name": "Titanium Ti-6Al-4V (Grade 5)",
                "properties": {
                    "thermal_conductivity": 6.7,
                    "specific_heat": 526.0,
                    "melting_point": 1660.0,
                    "density": 4430.0,
                    "thermal_expansion": 8.6e-6,
                    "youngs_modulus": 113.8e9,
                    "yield_strength": 880e6,
                    "max_service_temp": "400 °C"
                },
                "tags": ["aerospace", "titanium", "high-strength", "standard"],
                "source": "NASA / MatWeb"
            },
            {
                "name": "Inconel 718",
                "properties": {
                    "thermal_conductivity": 11.4,
                    "specific_heat": 435.0,
                    "melting_range": "1260-1336 °C",
                    "density": 8190.0,
                    "thermal_expansion": 13.0e-6,
                    "youngs_modulus": 200e9,
                    "yield_strength": 1100e6,
                    "max_service_temp": "700 °C"
                },
                "tags": ["superalloy", "nickel", "high-temperature", "rocket-engines"],
                "source": "Special Metals Corp / NASA"
            },
            {
                "name": "Tungsten (Pure)",
                "properties": {
                    "thermal_conductivity": 173.0,
                    "specific_heat": 132.0,
                    "melting_point": 3422.0,
                    "density": 19300.0,
                    "thermal_expansion": 4.5e-6,
                    "youngs_modulus": 411e9,
                    "max_service_temp": "2500 °C (Vacuum)"
                },
                "tags": ["refractory", "extreme-temperature", "radiation-shielding"],
                "source": "NIST Standard Reference Materials"
            },
             {
                "name": "Molybdenum (Pure)",
                "properties": {
                    "thermal_conductivity": 138.0,
                    "specific_heat": 250.0,
                    "melting_point": 2623.0,
                    "density": 10280.0,
                    "thermal_expansion": 4.8e-6,
                    "youngs_modulus": 329e9,
                    "max_service_temp": "1900 °C"
                },
                "tags": ["refractory", "high-temperature", "aerospace"],
                "source": "NIST"
            },
            {
                "name": "Silicon Carbide (SiC)",
                "properties": {
                    "thermal_conductivity": 120.0,
                    "specific_heat": 750.0,
                    "melting_point": 2730.0, # Decomposition
                    "density": 3210.0,
                    "thermal_expansion": 4.0e-6,
                    "youngs_modulus": 410e9,
                    "compressive_strength": 3900e6
                },
                "tags": ["ceramic", "high-temperature", "heat-shields", "armor"],
                "source": "NASA / Ceramic Source"
            },
            {
                "name": "Carbon-Carbon Composite",
                "properties": {
                    "thermal_conductivity": 50.0, # Anisotropic
                    "specific_heat": 710.0,
                    "sublimation_point": "3650 °C",
                    "density": 1800.0,
                    "thermal_expansion": 1.0e-6,
                    "youngs_modulus": 70e9
                },
                "tags": ["composite", "ultra-high-temp", "heat-shields", "nozzles"],
                "source": "NASA Space Shuttle Materials"
            }
        ]
        
        for mat in materials:
            print(f"Ingesting {mat['name']}...")
            self.db.upsert_material(
                material_name=mat['name'],
                properties=mat['properties'],
                source_url=mat['source'],
                tags=mat['tags']
            )

if __name__ == "__main__":
    ingestor = AerospaceMaterialsIngestor()
    ingestor.run()
