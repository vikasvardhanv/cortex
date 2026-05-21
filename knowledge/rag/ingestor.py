import os
from knowledge.rag.vector_db import CortexVectorDB
from knowledge.rag.scrapers.engineering_toolbox import EngineeringToolboxScraper
from knowledge.rag.scrapers.aerospace_materials import AerospaceMaterialsIngestor
from knowledge.rag.scrapers.physics_knowledge import PhysicsEngineeringKnowledgeIngestor

class CortexIngestor:
    def __init__(self):
        self.db = CortexVectorDB()
        self.scrapers = [
            EngineeringToolboxScraper(),
            AerospaceMaterialsIngestor(),
            PhysicsEngineeringKnowledgeIngestor(),
        ]
        
        # Inject the shared DB into scrapers that need it
        for scraper in self.scrapers:
            scraper.db = self.db

    def run_all(self):
        print("Starting global knowledge ingestion...")
        for scraper in self.scrapers:
            print(f"Running {scraper.__class__.__name__}...")
            try:
                # EngineeringToolboxScraper has specific methods
                if isinstance(scraper, EngineeringToolboxScraper):
                    scraper.scrape_metals_thermal_conductivity()
                else:
                    scraper.run()
            except Exception as e:
                print(f"Scraper {scraper.__class__.__name__} failed: {e}")

if __name__ == "__main__":
    ingestor = CortexIngestor()
    ingestor.run_all()
