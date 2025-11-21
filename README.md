Welcome to the public and hopefully official repository for the course project of the course 'Algorithm Analysis and Design'!
Team Name: Queue-ties

All the team members pelase use separate branches for your work and separate files for each algorithm to ensure a clean work flow and avoid merge conflicts :D

Cheers! Happy Coding!

---

## Project Links

**GUIDELINES:**  
https://docs.google.com/document/d/1lrng4n-TDOa2FGGMzhuY09IXE74mPiczbxdr6F74zfY/edit?usp=sharing

**PROBLEM STATEMENT:**  
https://docs.google.com/document/d/1RuMr-KCwL8jYXYH68ogDT0ZZxvLw33FtoMN3c2GChSI/edit?usp=sharing

**PLAN:**  
https://docs.google.com/document/d/1b_kc7eKNTcnQBP71B1Ox8n9kamsTorEdO3yrt3FOZg4/edit?tab=t.0

---

## Quick Start

### Setup

```bash
# Clone repository
git clone <repo-url>
cd course-project-queue-ties

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows Git Bash)
. venv/Scripts/activate
# Or on macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Components

```bash
# Data generation and graph construction (Guntesh)
python src/data_generation.py

# Union-Find (Guntesh)
python src/union_find.py

# BFS (Guntesh)
python src/bfs.py

# Girvan-Newman (Avani)
python girvan_newman.py

# Add other algorithms as they're implemented...
```

### Run Tests

```bash
# All tests
pytest

# Specific algorithm tests
pytest tests/test_union_find.py
pytest tests/test_bfs.py
pytest tests/test_girvan_newman.py
```

### Run Benchmarks

```bash
# See USAGE_GUIDE.md for detailed instructions
python benchmarks/benchmark_yourname.py
python visualize_results.py
```

---

## Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - How to benchmark your algorithm and generate visualizations
- **[TESTCASES.md](TESTCASES.md)** - Standardized test cases for all algorithms
- **[PLAN.md](PLAN.md)** - Task assignments and timeline

---

## Repository Structure

```
course-project-queue-ties/
├── src/                          # Shared modules (main branch)
│   ├── data_generation.py        # Stock data generator (Guntesh)
│   └── graph.py                  # Graph representation (Guntesh)
├── benchmarks/                   # Benchmark scripts
├── tests/                        # Unit tests
├── results/                      # Benchmark results (gitignored)
├── visualize_results.py          # Dynamic visualization script
├── USAGE_GUIDE.md                # Usage instructions
├── TESTCASES.md                  # Standardized test cases
└── requirements.txt              # Python dependencies
```

---

## Team Members & Algorithms

- **Guntesh Singh** - Data Foundation, Union-Find, BFS
- **Avani Sood** - Girvan-Newman
- **Khushi Dhingra** - DFS, PageRank
- **Saanvi Jain** - Louvain
- **Vaibhavi Kolipaka** - Node2Vec, Recommendations

