# CORTEX CEM

**Computational ORchestration for Technical Engineering eXecution**

Cortex is an AI-powered computational engineering model (CEM). It combines physics solvers, machine learning, and optimization to generate optimized engineering designs from natural language descriptions.

## What is a CEM?

A Computational Engineering Model is not a single thing like an LLM. It's a broad term for any model that uses computation to simulate, predict, or solve engineering problems — think physics simulations, structural analysis, fluid dynamics, optimization, etc.

Cortex orchestrates multiple computational methods:
- **Physics-based solvers** — Finite Element Method (FEM), Finite Difference, CFD
- **Data-driven models** — ML/DL surrogate models for fast predictions
- **Hybrid models** — Physics-Informed Neural Networks (PINNs)
- **Optimization engines** — Genetic algorithms, gradient-based optimization
- **Multi-model pipelines** — Chaining models together

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CORTEX CEM                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  LAYER 1: ROUTER (LLM-powered)                        │ │
│  │  "Design a heat shield" → ProblemSpec                 │ │
│  └───────────────────────────────────────────────────────┘ │
│                          │                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  LAYER 2: KNOWLEDGE BASE (Qdrant RAG)                 │ │
│  │  • 100+ Engineering Principles (NASA Technical)       │ │
│  │  • Real-world Material Database (Inconel, Titanium)    │ │
│  │  • Automated Scrapers (Engineering ToolBox)           │ │
│  └───────────────────────────────────────────────────────┘ │
│                          │                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  LAYER 3: SOLVERS                                     │ │
│  │  • ThermalSolver (Steady-state + Transient)           │ │
│  │  • StructuralSolver (Coming Soon)                     │ │
│  └───────────────────────────────────────────────────────┘ │
│                          │                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  LAYER 4: GEOMETRY ENGINE (SDF Core)                  │ │
│  │  • Parametric Primitives (Box, Cylinder, Sphere)       │ │
│  │  • Voxelization & Meshing (Marching Cubes)            │ │
│  │  • OBJ Final Export                                   │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
cd Cortex
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Configure API Key
### 3. Launch UI Hub (Chatbot Interface)
```bash
# In your terminal (with venv activated)
streamlit run app.py
```

### 4. Run Physics Examples
...

Create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

### 3. Run Tests

```bash
python examples/quick_test.py
```

### 4. Run Examples

```bash
python examples/heat_conduction_plate.py
```

## Usage

### Natural Language Input

```python
from cortex import CortexEngine

engine = CortexEngine()

# Describe your problem in natural language
result = engine.run(
    "Design a heat sink for a 50W LED. "
    "The base is 5cm x 5cm aluminum, cooled by natural convection."
)

# View results
print(result.summary())
result.plot_temperature()
```

### Direct API

```python
from solvers import ThermalSolver, ThermalProblem
from solvers.thermal import BoundaryCondition
from knowledge import MaterialsDB

# Get material
materials = MaterialsDB()
aluminum = materials.get("aluminum_6061")

# Define problem
problem = ThermalProblem(
    name="Heat Sink Analysis",
    domain_size=(0.05, 0.05),
    grid_size=(50, 50),
    material=aluminum,
    boundary_conditions={
        "bottom": BoundaryCondition("dirichlet", 353.0),  # 80°C heat source
        "top": BoundaryCondition("convection", 293.0, coefficient=25.0),  # Air cooling
        "left": BoundaryCondition("neumann", 0.0),  # Insulated
        "right": BoundaryCondition("neumann", 0.0),
    },
)

# Solve
solver = ThermalSolver()
result = solver.solve(problem)
print(result.summary())
```

### Intelligence & Knowledge Base

Cortex uses a **RAG (Retrieval-Augmented Generation)** system backed by **Qdrant** to ground its engineering decisions.

- **Physics Knowledge**: Contains rocket equations ($I_{sp}$, Thrust), thermodynamics, fluid dynamics (Reynolds, Mach), and structural mechanics.
- **Materials**: Real-world property data for Aluminum 6061-T6, Titanium Grade 5, Inconel 718, Tungsten, and Silicon Carbide.
- **Scrapers**: Automated integration with external databases like Engineering ToolBox.

**Metals:**
- aluminum_6061, stainless_steel_316, inconel_718, titanium_ti6al4v, copper_c11000, tungsten, molybdenum

**Ceramics & High-Temp:**
- silicon_carbide, alumina_al2o3, carbon_carbon, reinforced_carbon_carbon, pica

```python
from knowledge import MaterialsDB

db = MaterialsDB()

# List all materials
print(db.list_all())

# Find high-temperature materials
high_temp = db.find_by_max_temp(1500)  # Materials that work above 1500K

# Get material properties
inconel = db.get("inconel_718")
print(f"Thermal conductivity: {inconel.thermal_conductivity} W/(m·K)")
print(f"Max service temp: {inconel.max_service_temp} K")
```

## Project Structure

```
Cortex/
├── cortex/                    # Main CEM package
│   ├── knowledge/             # Engineering knowledge base
│   │   ├── rag/               # Qdrant RAG system
│   │   │   ├── vector_db.py   # Vector DB interface
│   │   │   ├── ingestor.py    # Global data ingestor
│   │   │   └── scrapers/      # Automated data scrapers
│   │   ├── materials.py       # Materials database
│   │   ├── physics_rules.py   # Physics equations
│   │   └── design_patterns.py # Design best practices
│   │
│   ├── solvers/               # Physics solvers
│   │   ├── thermal.py         # Heat conduction solver
│   │   └── base.py            # Base solver interface
│   │
│   ├── pipeline/              # Orchestration
│   │   ├── router.py          # LLM-powered problem parser
│   │   ├── executor.py        # Pipeline execution
│   │   └── engine.py          # Main CortexEngine
│   │
│   └── geometry/              # Geometry engine (SDFs, voxels, meshing)
│       ├── core/              # SDF, CSG, Voxel, Mesh
│       └── shapes/            # Primitives, Lattice, Profile
│
├── examples/                  # Example scripts
│   ├── quick_test.py          # Verification script
│   ├── heat_conduction_plate.py # 2D plate analysis
│   └── create_something.py     # Prompt-to-3D creation flow
│
└── .env                       # API keys (not in git)
```

## Roadmap

### Phase 1: Thermal Analysis ✅
- [x] Finite difference thermal solver (Steady + Transient)
- [x] RAG-enhanced Knowledge Base (Qdrant)
- [x] LLM-powered problem parsing with Knowledge injection

### Phase 2: Structural Analysis
- [ ] FEM structural solver (SfePy wrapper)
- [ ] Coupled thermal-structural analysis
- [ ] Stress validation

### Phase 3: Geometry Generation ✅
- [x] SDF-based parametric geometry (Primitives)
- [x] Voxelization & Meshing (Marching Cubes)
- [x] OBJ Export for 3D printing/analysis
- [ ] Lattice/gyroid structures

### Phase 4: Machine Learning
- [ ] Surrogate model training
- [ ] Physics-Informed Neural Networks (PINNs)
- [ ] Fast prediction mode

## Inspiration

This project is inspired by:
- [Leap71 Noyron](https://leap71.com/) - Computational Engineering Models
- [PicoGK](https://github.com/leap71/PicoGK) - Geometry kernel for computational engineering

## License

MIT License
