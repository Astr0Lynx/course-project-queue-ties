"""
Visualization Script for Benchmark Results
Author: Guntesh Singh
Description: Generate charts and graphs from benchmark data for report and presentation
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os


def load_benchmark_results(filepath='results/guntesh_benchmarks.json'):
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_runtime_comparison(results, output_dir='results'):
    """
    Plot runtime comparison between Union-Find and BFS.
    """
    # Separate by algorithm
    uf_results = [r for r in results if r['algorithm'] == 'Union-Find']
    bfs_results = [r for r in results if r['algorithm'] == 'BFS']
    
    # Group by size
    sizes = sorted(set(r['num_stocks'] for r in uf_results))
    scenarios = ['stable', 'normal', 'volatile', 'crash']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Union-Find runtime
    for scenario in scenarios:
        runtimes = []
        for size in sizes:
            matching = [r for r in uf_results 
                       if r['num_stocks'] == size and r['scenario'] == scenario]
            if matching:
                runtimes.append(matching[0]['runtime_seconds'])
            else:
                runtimes.append(0)
        ax1.plot(sizes, runtimes, marker='o', label=scenario.capitalize(), linewidth=2)
    
    ax1.set_xlabel('Number of Stocks', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Union-Find Runtime Performance', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot BFS runtime
    for scenario in scenarios:
        runtimes = []
        for size in sizes:
            matching = [r for r in bfs_results 
                       if r['num_stocks'] == size and r['scenario'] == scenario]
            if matching:
                runtimes.append(matching[0]['runtime_seconds'])
            else:
                runtimes.append(0)
        ax2.plot(sizes, runtimes, marker='s', label=scenario.capitalize(), linewidth=2)
    
    ax2.set_xlabel('Number of Stocks', fontsize=12)
    ax2.set_ylabel('Runtime (seconds)', fontsize=12)
    ax2.set_title('BFS Runtime Performance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/runtime_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/runtime_comparison.png")
    plt.close()


def plot_components_analysis(results, output_dir='results'):
    """
    Plot number of connected components across scenarios.
    """
    uf_results = [r for r in results if r['algorithm'] == 'Union-Find']
    
    sizes = sorted(set(r['num_stocks'] for r in uf_results))
    scenarios = ['stable', 'normal', 'volatile', 'crash']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(sizes))
    width = 0.2
    
    for i, scenario in enumerate(scenarios):
        components = []
        for size in sizes:
            matching = [r for r in uf_results 
                       if r['num_stocks'] == size and r['scenario'] == scenario]
            if matching:
                components.append(matching[0]['num_components'])
            else:
                components.append(0)
        
        ax.bar(x + i * width, components, width, 
               label=scenario.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Graph Size (Number of Stocks)', fontsize=12)
    ax.set_ylabel('Number of Connected Components', fontsize=12)
    ax.set_title('Market Segmentation: Connected Components by Scenario', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/components_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/components_analysis.png")
    plt.close()


def plot_path_length_analysis(results, output_dir='results'):
    """
    Plot average path lengths from BFS analysis.
    """
    bfs_results = [r for r in results if r['algorithm'] == 'BFS']
    
    sizes = sorted(set(r['num_stocks'] for r in bfs_results))
    scenarios = ['stable', 'normal', 'volatile', 'crash']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for scenario in scenarios:
        path_lengths = []
        for size in sizes:
            matching = [r for r in bfs_results 
                       if r['num_stocks'] == size and r['scenario'] == scenario]
            if matching:
                path_lengths.append(matching[0]['avg_path_length'])
            else:
                path_lengths.append(0)
        
        ax.plot(sizes, path_lengths, marker='o', 
                label=scenario.capitalize(), linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Stocks', fontsize=12)
    ax.set_ylabel('Average Shortest Path Length', fontsize=12)
    ax.set_title('BFS Analysis: Average Path Length by Scenario', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/path_length_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/path_length_analysis.png")
    plt.close()


def plot_graph_density(results, output_dir='results'):
    """
    Plot graph density across different scenarios.
    """
    uf_results = [r for r in results if r['algorithm'] == 'Union-Find']
    
    sizes = sorted(set(r['num_stocks'] for r in uf_results))
    scenarios = ['stable', 'normal', 'volatile', 'crash']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for scenario in scenarios:
        densities = []
        for size in sizes:
            matching = [r for r in uf_results 
                       if r['num_stocks'] == size and r['scenario'] == scenario]
            if matching:
                r = matching[0]
                # Calculate density
                n = r['num_stocks']
                e = r['num_edges']
                density = (2 * e) / (n * (n - 1)) if n > 1 else 0
                densities.append(density)
            else:
                densities.append(0)
        
        ax.plot(sizes, densities, marker='D', 
                label=scenario.capitalize(), linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Stocks', fontsize=12)
    ax.set_ylabel('Graph Density', fontsize=12)
    ax.set_title('Graph Density by Market Scenario', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/graph_density.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/graph_density.png")
    plt.close()


def plot_scalability(results, output_dir='results'):
    """
    Plot scalability: runtime vs graph size and edges.
    """
    uf_results = [r for r in results if r['algorithm'] == 'Union-Find']
    bfs_results = [r for r in results if r['algorithm'] == 'BFS']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Union-Find: runtime vs edges
    edges = [r['num_edges'] for r in uf_results]
    runtimes = [r['runtime_seconds'] * 1000 for r in uf_results]  # Convert to ms
    colors = [{'stable': 'blue', 'normal': 'green', 
               'volatile': 'orange', 'crash': 'red'}[r['scenario']] 
              for r in uf_results]
    
    ax1.scatter(edges, runtimes, c=colors, alpha=0.6, s=100)
    ax1.set_xlabel('Number of Edges', fontsize=12)
    ax1.set_ylabel('Runtime (milliseconds)', fontsize=12)
    ax1.set_title('Union-Find Scalability', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Stable'),
                      Patch(facecolor='green', label='Normal'),
                      Patch(facecolor='orange', label='Volatile'),
                      Patch(facecolor='red', label='Crash')]
    ax1.legend(handles=legend_elements)
    
    # BFS: runtime vs edges
    edges = [r['num_edges'] for r in bfs_results]
    runtimes = [r['runtime_seconds'] * 1000 for r in bfs_results]  # Convert to ms
    colors = [{'stable': 'blue', 'normal': 'green', 
               'volatile': 'orange', 'crash': 'red'}[r['scenario']] 
              for r in bfs_results]
    
    ax2.scatter(edges, runtimes, c=colors, alpha=0.6, s=100)
    ax2.set_xlabel('Number of Edges', fontsize=12)
    ax2.set_ylabel('Runtime (milliseconds)', fontsize=12)
    ax2.set_title('BFS Scalability', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scalability_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/scalability_analysis.png")
    plt.close()


def plot_component_size_distribution(results, output_dir='results'):
    """
    Plot distribution of component sizes for largest graph.
    """
    uf_results = [r for r in results if r['algorithm'] == 'Union-Find']
    
    # Get largest graph for each scenario (200 stocks)
    scenarios = ['stable', 'normal', 'volatile', 'crash']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, scenario in enumerate(scenarios):
        matching = [r for r in uf_results 
                   if r['num_stocks'] == 200 and r['scenario'] == scenario]
        
        if matching and matching[0]['component_sizes']:
            sizes = matching[0]['component_sizes']
            
            # Create histogram
            axes[i].hist(sizes, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            axes[i].set_xlabel('Component Size', fontsize=11)
            axes[i].set_ylabel('Frequency', fontsize=11)
            axes[i].set_title(f'{scenario.capitalize()} Market - Component Distribution', 
                            fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            avg_size = np.mean(sizes)
            max_size = max(sizes)
            axes[i].axvline(avg_size, color='red', linestyle='--', 
                          label=f'Avg: {avg_size:.1f}')
            axes[i].axvline(max_size, color='green', linestyle='--', 
                          label=f'Max: {max_size}')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/component_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/component_distribution.png")
    plt.close()


def create_summary_table(results, output_dir='results'):
    """
    Create a summary table image.
    """
    uf_results = [r for r in results if r['algorithm'] == 'Union-Find']
    bfs_results = [r for r in results if r['algorithm'] == 'BFS']
    
    # Prepare data
    scenarios = ['stable', 'normal', 'volatile', 'crash']
    sizes = [50, 100, 200]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Union-Find table
    uf_data = []
    for scenario in scenarios:
        row = [scenario.capitalize()]
        for size in sizes:
            matching = [r for r in uf_results 
                       if r['num_stocks'] == size and r['scenario'] == scenario]
            if matching:
                r = matching[0]
                row.append(f"{r['num_components']}\n({r['runtime_seconds']*1000:.2f}ms)")
            else:
                row.append("N/A")
        uf_data.append(row)
    
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=uf_data,
                       colLabels=['Scenario', '50 stocks', '100 stocks', '200 stocks'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.25, 0.25, 0.25, 0.25])
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 2)
    
    # Style header
    for i in range(4):
        table1[(0, i)].set_facecolor('#4CAF50')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    ax1.set_title('Union-Find: Components (Runtime)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # BFS table
    bfs_data = []
    for scenario in scenarios:
        row = [scenario.capitalize()]
        for size in sizes:
            matching = [r for r in bfs_results 
                       if r['num_stocks'] == size and r['scenario'] == scenario]
            if matching:
                r = matching[0]
                row.append(f"{r['avg_path_length']:.2f}\n({r['runtime_seconds']*1000:.2f}ms)")
            else:
                row.append("N/A")
        bfs_data.append(row)
    
    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(cellText=bfs_data,
                       colLabels=['Scenario', '50 stocks', '100 stocks', '200 stocks'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.25, 0.25, 0.25, 0.25])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2)
    
    # Style header
    for i in range(4):
        table2[(0, i)].set_facecolor('#2196F3')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title('BFS: Avg Path Length (Runtime)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/summary_table.png")
    plt.close()


def main():
    """Generate all visualizations."""
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Create output directory
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print("\nLoading benchmark results...")
    results = load_benchmark_results()
    print(f"✓ Loaded {len(results)} benchmark results")
    
    # Generate all plots
    print("\nGenerating charts...")
    plot_runtime_comparison(results, output_dir)
    plot_components_analysis(results, output_dir)
    plot_path_length_analysis(results, output_dir)
    plot_graph_density(results, output_dir)
    plot_scalability(results, output_dir)
    plot_component_size_distribution(results, output_dir)
    create_summary_table(results, output_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated {7} visualization files in '{output_dir}/':")
    print("  1. runtime_comparison.png       - Algorithm runtime comparison")
    print("  2. components_analysis.png       - Connected components analysis")
    print("  3. path_length_analysis.png      - BFS path length analysis")
    print("  4. graph_density.png             - Graph density by scenario")
    print("  5. scalability_analysis.png      - Scalability vs edges")
    print("  6. component_distribution.png    - Component size distributions")
    print("  7. summary_table.png             - Summary table")
    print("\nThese can be used in your report and presentation!")
    print("="*70)


if __name__ == "__main__":
    main()
