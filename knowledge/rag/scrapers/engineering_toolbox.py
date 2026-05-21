import requests
from bs4 import BeautifulSoup
import pandas as pd
from knowledge.rag.vector_db import CortexVectorDB
import time

class EngineeringToolboxScraper:
    def __init__(self):
        self.base_url = "https://www.engineeringtoolbox.com"
        self.db = CortexVectorDB()

    def scrape_metals_thermal_conductivity(self):
        url = f"{self.base_url}/thermal-conductivity-metals-d_858.html"
        print(f"Scraping {url}...")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Engineering ToolBox often puts tables in specific divs
        tables = soup.find_all('table')
        if not tables:
            print("No tables found")
            return

        # Target the main data table
        df = pd.read_html(str(tables[0]))[0]
        
        # Clean up columns (Engineering toolbox tables can be messy)
        # Usually: Metal | Conductivity (W/mK) | Temperature (°C)
        print(f"Found {len(df)} entries")
        
        for idx, row in df.iterrows():
            try:
                # Basic cleaning - adjust based on actual table structure
                metal_name = str(row.iloc[0])
                conductivity = str(row.iloc[1])
                
                if "Metal" in metal_name or "Thermal" in metal_name:
                    continue

                props = {
                    "thermal_conductivity": conductivity,
                    "unit": "W/mK",
                    "category": "Metal"
                }
                
                self.db.upsert_material(
                    material_name=metal_name,
                    properties=props,
                    source_url=url,
                    tags=["thermal", "conductivity", "metals"]
                )
                print(f"Uploaded: {metal_name}")
            except Exception as e:
                print(f"Error processing row {idx}: {e}")

if __name__ == "__main__":
    scraper = EngineeringToolboxScraper()
    scraper.scrape_metals_thermal_conductivity()
