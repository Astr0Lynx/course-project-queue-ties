# Stock Market Tangle - Team Queue-ties

Welcome to the official repository for the **Stock Market Tangle** course project!

## Team Members

- **Guntesh Singh** - Data Foundation, Union-Find, BFS
- **Khushi Dhingra** - DFS, PageRank, Disruption Simulations
- **Avani Sood** - Girvan-Newman Algorithm
- **Saanvi Jain** - Louvain Algorithm
- **Vaibhavi Kolipaka** - Node2Vec, Recommendations, Integration

## Project Overview

This project analyzes stock market correlations using graph-based algorithms to:
- Identify market segments (connected components)
- Find correlation chains (shortest paths)
- Detect influential stocks (centrality)
- Discover sector clusters (community detection)
- Generate portfolio recommendations (embeddings)

## Repository Structure

```
course-project-queue-ties/
├── src/                    # Source code (each member has their files)
│   ├── data_generation.py  # [Guntesh] Data generation
│   ├── graph.py            # [Guntesh] Shared graph structure
│   ├── union_find.py       # [Guntesh] Union-Find algorithm
│   ├── bfs.py              # [Guntesh] BFS algorithm
│   ├── dfs.py              # [Khushi] DFS algorithm (to be added)
│   ├── pagerank.py         # [Khushi] PageRank algorithm (to be added)
│   ├── girvan_newman.py    # [Avani] Girvan-Newman (to be added)
│   ├── louvain.py          # [Saanvi] Louvain algorithm (to be added)
│   └── node2vec.py         # [Vaibhavi] Node2Vec (to be added)
├── tests/                  # Unit tests for each algorithm
├── benchmarks/             # Benchmarking scripts
├── data/                   # Generated datasets
├── results/                # Experimental results
├── docs/                   # Documentation and report sections
├── main.py                 # Main integration script
├── requirements.txt        # Python dependencies
├── PLAN.md                 # Team plan and task distribution
└── README.md               # This file
```

## Development Workflow

### Branch Strategy

**IMPORTANT:** Each team member works on their own branch!

- `main` - Protected branch (requires PR for merges)
- `guntesh-data-foundation` - Guntesh's work ✅
- `khushi-traversal-centrality` - Khushi's work
- `avani-girvan-newman` - Avani's work
- `saanvi-louvain` - Saanvi's work
- `vaibhavi-node2vec` - Vaibhavi's work

### Getting Started

1. **Clone the repository** (if not already done)

2. **Create your branch:**
   ```bash
   git checkout -b <your-branch-name>
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Work on your files** in the `src/` directory

5. **Commit frequently:**
   ```bash
   git add .
   git commit -m "Clear description of changes"
   git push origin <your-branch-name>
   ```

### Merge Guidelines

- **Never push directly to `main`**
- Create Pull Requests (PRs) when your feature is ready
- Request code review from at least one team member
- Ensure tests pass before merging
- Resolve conflicts before merging

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip
- Git

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Complete Workflow

```bash
python main.py
```

### Run Individual Components

```bash
# Generate data (Guntesh)
cd src
python data_generation.py

# Run Union-Find (Guntesh)
python union_find.py

# Run BFS (Guntesh)
python bfs.py

# (Other members will add their commands here)
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
python tests/test_union_find.py
python tests/test_bfs.py
```

### Run Benchmarks

```bash
python benchmarks/benchmark_guntesh.py
# (Other benchmarking scripts to be added)
```

## Coding Standards

### Python Style
- Follow PEP 8 style guide
- Use type hints where possible
- Write docstrings for all functions/classes
- Maximum line length: 88 characters

### Documentation Requirements
- **Module docstring**: Brief description, author name
- **Function docstrings**: Purpose, Args, Returns, Time/Space complexity
- **Inline comments**: Explain complex logic

### Example:
```python
"""
Module description.
Author: Your Name
"""

def algorithm_name(graph, param: int) -> List[str]:
    """
    Brief description of what the algorithm does.
    
    Args:
        graph: Graph object
        param: Description of parameter
    
    Returns:
        Description of return value
    
    Time Complexity: O(...)
    Space Complexity: O(...)
    """
    # Implementation
    pass
```

### Implementation Rules

1. **No external graph libraries** for core algorithm logic
   - ❌ No NetworkX, igraph for algorithm implementations
   - ✅ Can use numpy, pandas for data manipulation
   - ✅ Can use basic Python data structures

2. **Modular code**
   - One file per algorithm
   - Shared utilities in separate files
   - Clear separation of concerns

3. **From-scratch implementations**
   - All graph algorithms must be manual implementations

## Shared Components

### Graph Structure (Provided by Guntesh)

```python
from graph import Graph

# Create graph
graph = Graph()
graph.add_node("STOCK_A", {"volatility": 0.02})
graph.add_edge("STOCK_A", "STOCK_B", weight=0.75)

# Access graph
neighbors = graph.get_neighbors("STOCK_A")
edges = graph.get_edges()
```

### Data Generation (Provided by Guntesh)

```python
from data_generation import StockDataGenerator

generator = StockDataGenerator(seed=42)
returns, corr_matrix, attributes = generator.generate_dataset(
    num_stocks=100,
    scenario="normal"
)
```

## Team Meetings

- **Weekly meetings**: 2-3 hours
- **Agenda**: Code integration, progress review, issue resolution
- **Rotating lead**: Coordinate and track tasks

## Timeline

- **Week 1**: Data generation, algorithm implementations ← **Current**
- **Week 2**: Simulations, metrics, visualizations
- **Week 3**: Comparisons, report drafting
- **Week 4**: Integration, testing, presentation prep

## Deliverables

### Code Deliverables
- ✅ All algorithms implemented from scratch
- ✅ Unit tests for each algorithm
- ✅ Benchmarking scripts
- ✅ Integration script (main.py)

### Report Deliverables
- Introduction (Guntesh)
- Algorithm descriptions (Each member for their algorithms)
- Experimental setup (Guntesh)
- Results & Analysis (Team effort, led by Vaibhavi)
- Conclusion (Vaibhavi)

### Presentation
- 15-minute presentation
- Each member presents their section
- Live demo of main.py

## Resources

- **Problem Statement**: `Stock Market Tangle.pdf`
- **Guidelines**: `AAD Final Project Guidelines.pdf`
- **Team Plan**: `PLAN.md`

### External Links
GUIDELINES:
https://docs.google.com/document/d/1lrng4n-TDOa2FGGMzhuY09IXE74mPiczbxdr6F74zfY/edit?usp=sharing

PROBLEM STATEMENT: 
https://docs.google.com/document/d/1RuMr-KCwL8jYXYH68ogDT0ZZxvLw33FtoMN3c2GChSI/edit?usp=sharing

PLAN:
https://docs.google.com/document/d/1b_kc7eKNTcnQBP71B1Ox8n9kamsTorEdO3yrt3FOZg4/edit?tab=t.0

## Troubleshooting

### Import Errors
Make sure you're in the project root and have installed dependencies:
```bash
pip install -r requirements.txt
```

### Git Conflicts
If you encounter merge conflicts:
1. Pull latest changes: `git pull origin main`
2. Resolve conflicts in your editor
3. Commit resolved changes
4. Push to your branch

### Test Failures
Run individual tests to identify issues:
```bash
python tests/test_<algorithm>.py -v
```

## Contributing

1. Create your feature branch
2. Write clean, documented code
3. Add tests for your algorithms
4. Update this README if needed
5. Submit a Pull Request

---

**All team members: Please use separate branches for your work and separate files for each algorithm to ensure a clean workflow and avoid merge conflicts!**

**Happy Coding, Team Queue-ties! 🚀**

