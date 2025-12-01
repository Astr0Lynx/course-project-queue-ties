Welcome to the official repository for the course project of 'Algorithm Analysis and Design'!

Team Name: Queue-ties
---

## Project Overview

This project implements various graph algorithms to solve the problem outlined in the [Problem Statement](https://docs.google.com/document/d/1RuMr-KCwL8jYXYH68ogDT0ZZxvLw33FtoMN3c2GChSI/edit?usp=sharing). The repository is organized to facilitate seamless navigation and evaluation.

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
python src/girvan_newman.py

# DFS (Khushi)
python src/dfs.py

# PageRank (Khushi)
python src/pagerank.py

# Louvain (Saanvi)
python src/louvain.py

# Node2Vec (Vaibhavi)
python src/node2vec.py

# Recommendations (Vaibhavi)
python src/recommendations.py
```

### Run Tests

```bash
# Run all tests using pytest
pytest
```

### Run Benchmarks
```bash
# Run benchmarks for all algorithms
python benchmarks/run_all_benchmarks.py
```
```bash
# See USAGE_GUIDE.md for detailed instructions
python visualize_results.py
```

---

## Documentation
- **[TESTCASES.md](TESTCASES.md)** - Standardized test cases for all algorithms

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

---

## Notes 

- Each algorithm is implemented in a separate file to ensure modularity.
- The repository includes detailed documentation for setup, usage, and testing.
- Please refer to the `USAGE_GUIDE.md` for instructions on running benchmarks and visualizing results.
- Test cases are standardized and can be found in `TESTCASES.md`.
- If you encounter any issues, feel free to reach out to the team members listed above.

