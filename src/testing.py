"""
Comprehensive Testing and Benchmarking for Stock Market Tangle Project
Author: Khushi Dhingra
Description: Tests DFS, PageRank, and Market Disruption algorithms using standardized test cases.
             Provides performance benchmarks and correctness validation.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json
from pathlib import Path

from .data_generation import StockDataGenerator
from .graph import build_graph_from_correlation
from .dfs import DFS, analyze_market_connectivity
from .pagerank import PageRank, identify_market_influencers
from .market_disruption import MarketDisruptionSimulator


class AlgorithmTester:
    """
    Comprehensive testing and benchmarking suite for all algorithms.
    Uses standardized test cases from TESTCASES.md for consistent results.
    """
    
    def __init__(self, save_results: bool = True, results_dir: str = "test_results"):
        """
        Initialize the testing suite.
        
        Args:
            save_results: Whether to save test results to files
            results_dir: Directory to save results
        """
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        if self.save_results:
            self.results_dir.mkdir(exist_ok=True)
        
        self.test_results = {}
        self.performance_metrics = {}
    
    def run_standardized_test_cases(self) -> Dict:
        """
        Run all standardized test cases from TESTCASES.md.
        
        Returns:
            Dictionary with comprehensive test results
        """
        print("Running standardized test cases...")
        
        # Test configuration from TESTCASES.md
        test_configs = [
            # Small graphs
            {"num_stocks": 50, "scenario": "stable", "threshold": 0.5},
            {"num_stocks": 50, "scenario": "normal", "threshold": 0.5},
            {"num_stocks": 50, "scenario": "volatile", "threshold": 0.5},
            {"num_stocks": 50, "scenario": "crash", "threshold": 0.3},
            
            # Medium graphs
            {"num_stocks": 100, "scenario": "stable", "threshold": 0.5},
            {"num_stocks": 100, "scenario": "normal", "threshold": 0.5},
            {"num_stocks": 100, "scenario": "volatile", "threshold": 0.5},
            {"num_stocks": 100, "scenario": "crash", "threshold": 0.3},
            
            # Large graphs
            {"num_stocks": 200, "scenario": "stable", "threshold": 0.5},
            {"num_stocks": 200, "scenario": "normal", "threshold": 0.5},
            {"num_stocks": 200, "scenario": "volatile", "threshold": 0.5},
            {"num_stocks": 200, "scenario": "crash", "threshold": 0.3},
        ]
        
        all_results = {}
        
        for i, config in enumerate(test_configs):
            print(f"Running test case {i+1}/{len(test_configs)}: {config}")
            
            # Generate test data
            generator = StockDataGenerator(seed=42)
            returns, corr_matrix, stock_attrs = generator.generate_dataset(
                config["num_stocks"], scenario=config["scenario"]
            )
            
            # Build graph
            graph = build_graph_from_correlation(
                corr_matrix, stock_attrs, config["threshold"]
            )
            
            # Run all algorithm tests
            test_name = f"{config['num_stocks']}_stocks_{config['scenario']}"
            all_results[test_name] = self.run_complete_algorithm_test(graph, stock_attrs, config)
        
        self.test_results["standardized_tests"] = all_results
        
        if self.save_results:
            self._save_results("standardized_test_results.json", all_results)
        
        return all_results
    
    def run_complete_algorithm_test(self, graph, stock_attributes: Dict, config: Dict) -> Dict:
        """
        Run complete test suite for all algorithms on a single graph.
        
        Args:
            graph: Stock correlation graph
            stock_attributes: Dictionary of stock attributes
            config: Test configuration
        
        Returns:
            Dictionary with test results for all algorithms
        """
        results = {
            "config": config,
            "graph_stats": {
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
                "density": (2 * graph.num_edges) / (graph.num_nodes * (graph.num_nodes - 1)) if graph.num_nodes > 1 else 0
            }
        }
        
        # Test DFS algorithms
        print("  Testing DFS algorithms...")
        results["dfs_tests"] = self.test_dfs_algorithms(graph, stock_attributes)
        
        # Test PageRank algorithm
        print("  Testing PageRank algorithm...")
        results["pagerank_tests"] = self.test_pagerank_algorithm(graph, stock_attributes)
        
        # Test Market Disruption simulations
        print("  Testing Market Disruption simulations...")
        results["disruption_tests"] = self.test_market_disruption(graph, stock_attributes)
        
        return results
    
    def test_dfs_algorithms(self, graph, stock_attributes: Dict) -> Dict:
        """
        Comprehensive testing of DFS implementation.
        
        Args:
            graph: Stock correlation graph
            stock_attributes: Dictionary of stock attributes
        
        Returns:
            Dictionary with DFS test results
        """
        dfs = DFS(graph)
        test_results = {}
        
        # Correctness tests
        nodes = graph.get_nodes()
        if nodes:
            start_node = nodes[0]
            
            # Test 1: Basic traversal correctness
            recursive_traversal = dfs.dfs_recursive(start_node)
            iterative_traversal = dfs.dfs_iterative(start_node)
            
            test_results["traversal_consistency"] = {
                "recursive_nodes_count": len(recursive_traversal),
                "iterative_nodes_count": len(iterative_traversal),
                "sets_equal": set(recursive_traversal) == set(iterative_traversal),
                "both_include_start": start_node in recursive_traversal and start_node in iterative_traversal
            }
            
            # Test 2: Path finding correctness
            if len(nodes) > 1:
                target_node = nodes[-1]
                recursive_path = dfs.find_path(start_node, target_node, "recursive")
                iterative_path = dfs.find_path(start_node, target_node, "iterative")
                
                test_results["path_finding"] = {
                    "recursive_path_exists": recursive_path is not None,
                    "iterative_path_exists": iterative_path is not None,
                    "recursive_path_valid": self._validate_path(graph, recursive_path) if recursive_path else False,
                    "iterative_path_valid": self._validate_path(graph, iterative_path) if iterative_path else False,
                    "paths_consistent": (recursive_path is None) == (iterative_path is None)
                }
            
            # Test 3: Connected components
            components = dfs.find_connected_components()
            total_nodes_in_components = sum(len(comp) for comp in components)
            
            test_results["connected_components"] = {
                "num_components": len(components),
                "total_nodes_coverage": total_nodes_in_components,
                "coverage_complete": total_nodes_in_components == len(nodes),
                "largest_component_size": max(len(comp) for comp in components) if components else 0
            }
            
            # Test 4: Connectivity check
            is_connected = dfs.is_connected()
            test_results["connectivity"] = {
                "is_connected": is_connected,
                "consistent_with_components": is_connected == (len(components) == 1)
            }
            
            # Test 5: Cycle detection
            has_cycle = dfs.has_cycle()
            test_results["cycle_detection"] = {
                "has_cycle": has_cycle,
                "cycle_expected": graph.num_edges >= graph.num_nodes  # Simple heuristic
            }
        
        # Performance tests
        if nodes:
            performance = dfs.benchmark_performance(start_node, num_runs=5)
            test_results["performance"] = performance
        
        # Market connectivity analysis
        connectivity_analysis = analyze_market_connectivity(graph, stock_attributes)
        test_results["market_connectivity"] = {
            "analysis_completed": True,
            "connectivity_ratio": connectivity_analysis.get("connectivity_ratio", 0),
            "sector_analysis_available": "sector_analysis" in connectivity_analysis
        }
        
        return test_results
    
    def test_pagerank_algorithm(self, graph, stock_attributes: Dict) -> Dict:
        """
        Comprehensive testing of PageRank implementation.
        
        Args:
            graph: Stock correlation graph
            stock_attributes: Dictionary of stock attributes
        
        Returns:
            Dictionary with PageRank test results
        """
        pagerank = PageRank(graph)
        test_results = {}
        
        # Correctness tests
        if graph.num_nodes > 0:
            # Test 1: Basic PageRank calculation
            scores = pagerank.calculate_pagerank(max_iterations=50, tolerance=1e-6)
            
            test_results["basic_calculation"] = {
                "scores_calculated": len(scores) > 0,
                "all_nodes_covered": len(scores) == graph.num_nodes,
                "scores_sum_normalized": abs(sum(scores.values()) - 1.0) < 1e-3,
                "all_scores_positive": all(score > 0 for score in scores.values()),
                "converged": pagerank.iterations < 50
            }
            
            # Test 2: Weighted vs unweighted comparison
            weighted_scores = pagerank.calculate_pagerank(weighted=True)
            unweighted_scores = pagerank.calculate_pagerank(weighted=False)
            
            correlation = np.corrcoef(
                list(weighted_scores.values()),
                list(unweighted_scores.values())
            )[0, 1] if len(weighted_scores) > 1 else 1.0
            
            test_results["weighted_comparison"] = {
                "correlation_coefficient": correlation,
                "significantly_different": abs(correlation) < 0.9,
                "weighted_converged": pagerank.iterations < 50
            }
            
            # Test 3: Convergence analysis
            convergence_info = pagerank.analyze_convergence()
            test_results["convergence"] = {
                "converged": convergence_info["converged"],
                "iterations": convergence_info["iterations_to_convergence"],
                "final_difference": convergence_info["final_difference"],
                "stable_convergence": convergence_info["final_difference"] < 1e-6
            }
            
            # Test 4: Top stocks identification
            top_stocks = pagerank.get_top_stocks(5)
            test_results["ranking"] = {
                "top_stocks_count": len(top_stocks),
                "scores_descending": all(top_stocks[i][1] >= top_stocks[i+1][1] 
                                       for i in range(len(top_stocks)-1)),
                "top_score_reasonable": top_stocks[0][1] > 1.0/graph.num_nodes if top_stocks else False
            }
            
            # Test 5: Centrality comparison
            centrality_comparison = pagerank.compare_with_degree_centrality()
            test_results["centrality_comparison"] = {
                "comparison_completed": "correlation_with_degree" in centrality_comparison,
                "correlation_with_degree": centrality_comparison.get("correlation_with_degree", 0),
                "hidden_influencers_found": len(centrality_comparison.get("hidden_influencers", [])) > 0
            }
        
        # Performance tests
        if graph.num_nodes > 0:
            performance = pagerank.benchmark_performance(max_iterations=50, num_runs=3)
            test_results["performance"] = performance
        
        # Market influence analysis
        influence_analysis = identify_market_influencers(graph, stock_attributes)
        test_results["market_influence"] = {
            "analysis_completed": True,
            "top_influencers_identified": len(influence_analysis.get("top_influencers", [])) > 0,
            "sector_analysis_available": "sector_influence" in influence_analysis
        }
        
        return test_results
    
    def test_market_disruption(self, graph, stock_attributes: Dict) -> Dict:
        """
        Test market disruption simulation capabilities.
        
        Args:
            graph: Stock correlation graph
            stock_attributes: Dictionary of stock attributes
        
        Returns:
            Dictionary with market disruption test results
        """
        simulator = MarketDisruptionSimulator(graph, stock_attributes)
        test_results = {}
        
        # Test 1: Market crash simulation
        try:
            crash_result = simulator.simulate_market_crash(severity=0.5)
            test_results["crash_simulation"] = {
                "simulation_completed": True,
                "scenario": crash_result.get("scenario"),
                "edges_added": crash_result.get("edges_added", 0),
                "connectivity_analyzed": "connectivity_changes" in crash_result,
                "influence_analyzed": "influence_changes" in crash_result
            }
        except Exception as e:
            test_results["crash_simulation"] = {"error": str(e), "simulation_completed": False}
        
        # Test 2: Sector collapse simulation
        sectors = set(attrs.get('sector', 'Unknown') for attrs in stock_attributes.values() 
                     if attrs.get('sector') != 'Unknown')
        
        if sectors:
            test_sector = list(sectors)[0]
            try:
                sector_result = simulator.simulate_sector_collapse(test_sector)
                test_results["sector_collapse"] = {
                    "simulation_completed": True,
                    "target_sector": sector_result.get("target_sector"),
                    "stocks_removed": len(sector_result.get("stocks_removed", [])),
                    "impact_analyzed": "connectivity_changes" in sector_result,
                    "systemic_risk_calculated": "systemic_risk_score" in sector_result
                }
            except Exception as e:
                test_results["sector_collapse"] = {"error": str(e), "simulation_completed": False}
        
        # Test 3: Key player removal simulation
        try:
            removal_result = simulator.simulate_key_player_removal("top_pagerank", 3)
            test_results["key_player_removal"] = {
                "simulation_completed": True,
                "strategy": removal_result.get("removal_strategy"),
                "players_removed": len(removal_result.get("players_removed", [])),
                "resilience_calculated": "network_resilience_score" in removal_result,
                "cascade_analyzed": "cascade_effects" in removal_result
            }
        except Exception as e:
            test_results["key_player_removal"] = {"error": str(e), "simulation_completed": False}
        
        # Test 4: Threshold sensitivity analysis
        try:
            threshold_result = simulator.simulate_correlation_threshold_changes([0.4, 0.5, 0.6])
            test_results["threshold_sensitivity"] = {
                "simulation_completed": True,
                "thresholds_tested": len(threshold_result.get("thresholds_tested", [])),
                "sensitivity_calculated": "sensitivity_metrics" in threshold_result,
                "optimal_found": "optimal_threshold" in threshold_result
            }
        except Exception as e:
            test_results["threshold_sensitivity"] = {"error": str(e), "simulation_completed": False}
        
        return test_results
    
    def run_performance_benchmarks(self) -> Dict:
        """
        Run comprehensive performance benchmarks for all algorithms.
        
        Returns:
            Dictionary with performance benchmark results
        """
        print("Running performance benchmarks...")
        
        # Test different graph sizes
        sizes = [50, 100, 200, 500]
        scenarios = ["normal"]
        
        benchmark_results = {}
        
        for size in sizes:
            print(f"Benchmarking with {size} stocks...")
            
            # Generate test data
            generator = StockDataGenerator(seed=42)
            returns, corr_matrix, stock_attrs = generator.generate_dataset(size, scenario="normal")
            graph = build_graph_from_correlation(corr_matrix, stock_attrs, 0.5)
            
            size_results = {}
            
            # Benchmark DFS
            dfs = DFS(graph)
            if graph.get_nodes():
                start_node = graph.get_nodes()[0]
                dfs_perf = dfs.benchmark_performance(start_node, num_runs=10)
                size_results["dfs"] = dfs_perf
            
            # Benchmark PageRank
            pagerank = PageRank(graph)
            if graph.num_nodes > 0:
                pr_perf = pagerank.benchmark_performance(max_iterations=100, num_runs=5)
                size_results["pagerank"] = pr_perf
            
            # Benchmark market disruption (limited to avoid long runtime)
            simulator = MarketDisruptionSimulator(graph, stock_attrs)
            start_time = time.time()
            try:
                simulator.simulate_market_crash(severity=0.5)
                disruption_time = time.time() - start_time
                size_results["market_disruption"] = {"avg_time": disruption_time}
            except Exception as e:
                size_results["market_disruption"] = {"error": str(e)}
            
            benchmark_results[f"{size}_stocks"] = size_results
        
        self.performance_metrics = benchmark_results
        
        if self.save_results:
            self._save_results("performance_benchmarks.json", benchmark_results)
        
        return benchmark_results
    
    def run_correctness_validation(self) -> Dict:
        """
        Run correctness validation tests for all algorithms.
        
        Returns:
            Dictionary with correctness validation results
        """
        print("Running correctness validation...")
        
        validation_results = {}
        
        # Create simple test graphs with known properties
        test_graphs = self._create_validation_graphs()
        
        for graph_name, (graph, stock_attrs, expected) in test_graphs.items():
            print(f"Validating on {graph_name}...")
            
            graph_results = {}
            
            # Validate DFS
            dfs = DFS(graph)
            if graph.get_nodes():
                # Test connectivity
                is_connected = dfs.is_connected()
                components = dfs.find_connected_components()
                
                graph_results["dfs"] = {
                    "connectivity_correct": is_connected == expected["is_connected"],
                    "component_count_correct": len(components) == expected["num_components"],
                    "cycle_detection": dfs.has_cycle() == expected["has_cycle"]
                }
            
            # Validate PageRank
            pagerank = PageRank(graph)
            if graph.num_nodes > 0:
                scores = pagerank.calculate_pagerank()
                
                # Check basic properties
                score_sum = sum(scores.values())
                all_positive = all(score > 0 for score in scores.values())
                
                graph_results["pagerank"] = {
                    "scores_sum_normalized": abs(score_sum - 1.0) < 1e-3,
                    "all_scores_positive": all_positive,
                    "convergence_achieved": pagerank.iterations < 100
                }
            
            validation_results[graph_name] = graph_results
        
        if self.save_results:
            self._save_results("correctness_validation.json", validation_results)
        
        return validation_results
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate a comprehensive test report combining all results.
        
        Returns:
            Dictionary with comprehensive test report
        """
        print("Generating comprehensive test report...")
        
        report = {
            "test_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tests_run": list(self.test_results.keys()),
                "performance_benchmarks_available": bool(self.performance_metrics)
            },
            "algorithm_performance": self._analyze_algorithm_performance(),
            "scalability_analysis": self._analyze_scalability(),
            "correctness_summary": self._summarize_correctness(),
            "recommendations": self._generate_test_recommendations()
        }
        
        if self.save_results:
            self._save_results("comprehensive_report.json", report)
            self._generate_performance_plots()
        
        return report
    
    # Helper methods
    def _validate_path(self, graph, path: List[str]) -> bool:
        """Validate that a path is correct."""
        if not path or len(path) < 2:
            return len(path) <= 1  # Single node or empty path is valid
        
        for i in range(len(path) - 1):
            if not graph.has_edge(path[i], path[i + 1]):
                return False
        
        return True
    
    def _create_validation_graphs(self) -> Dict:
        """Create simple graphs with known properties for validation."""
        from .graph import Graph
        
        graphs = {}
        
        # Create disconnected graph
        disconnected = Graph()
        for i in range(5):
            disconnected.add_node(f"stock_{i}")
        disconnected.add_edge("stock_0", "stock_1")
        disconnected.add_edge("stock_2", "stock_3")
        
        graphs["disconnected"] = (disconnected, {}, {
            "is_connected": False,
            "num_components": 3,  # stock_0-stock_1, stock_2-stock_3, stock_4
            "has_cycle": False
        })
        
        # Create connected graph with cycle
        connected_cycle = Graph()
        for i in range(4):
            connected_cycle.add_node(f"stock_{i}")
        connected_cycle.add_edge("stock_0", "stock_1")
        connected_cycle.add_edge("stock_1", "stock_2")
        connected_cycle.add_edge("stock_2", "stock_3")
        connected_cycle.add_edge("stock_3", "stock_0")
        
        graphs["connected_cycle"] = (connected_cycle, {}, {
            "is_connected": True,
            "num_components": 1,
            "has_cycle": True
        })
        
        return graphs
    
    def _analyze_algorithm_performance(self) -> Dict:
        """Analyze performance characteristics of algorithms."""
        if not self.performance_metrics:
            return {"error": "No performance data available"}
        
        analysis = {}
        
        # Analyze DFS performance
        dfs_times = []
        sizes = []
        for size_key, results in self.performance_metrics.items():
            if "dfs" in results and "recursive_avg_time" in results["dfs"]:
                size = int(size_key.split("_")[0])
                sizes.append(size)
                dfs_times.append(results["dfs"]["recursive_avg_time"])
        
        if len(sizes) > 1:
            analysis["dfs"] = {
                "time_complexity_trend": self._estimate_complexity(sizes, dfs_times),
                "avg_time_per_size": dict(zip(sizes, dfs_times))
            }
        
        # Analyze PageRank performance
        pr_times = []
        pr_sizes = []
        for size_key, results in self.performance_metrics.items():
            if "pagerank" in results and "weighted_avg_time" in results["pagerank"]:
                size = int(size_key.split("_")[0])
                pr_sizes.append(size)
                pr_times.append(results["pagerank"]["weighted_avg_time"])
        
        if len(pr_sizes) > 1:
            analysis["pagerank"] = {
                "time_complexity_trend": self._estimate_complexity(pr_sizes, pr_times),
                "avg_time_per_size": dict(zip(pr_sizes, pr_times))
            }
        
        return analysis
    
    def _estimate_complexity(self, sizes: List[int], times: List[float]) -> str:
        """Estimate time complexity from size vs time data."""
        if len(sizes) < 2:
            return "insufficient_data"
        
        # Check different complexity patterns
        ratios = []
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1] if times[i-1] > 0 else float('inf')
            ratios.append(time_ratio / size_ratio)
        
        avg_ratio = np.mean(ratios)
        
        if avg_ratio < 1.5:
            return "O(n) - Linear"
        elif avg_ratio < 3:
            return "O(n log n) - Linearithmic"
        elif avg_ratio < 5:
            return "O(n^2) - Quadratic"
        else:
            return "O(n^k) - Polynomial or worse"
    
    def _analyze_scalability(self) -> Dict:
        """Analyze how algorithms scale with graph size."""
        if not self.performance_metrics:
            return {"error": "No performance data available"}
        
        scalability = {}
        
        sizes = []
        dfs_times = []
        pr_times = []
        
        for size_key, results in self.performance_metrics.items():
            size = int(size_key.split("_")[0])
            sizes.append(size)
            
            dfs_time = results.get("dfs", {}).get("recursive_avg_time", 0)
            pr_time = results.get("pagerank", {}).get("weighted_avg_time", 0)
            
            dfs_times.append(dfs_time)
            pr_times.append(pr_time)
        
        if sizes:
            scalability = {
                "size_range_tested": f"{min(sizes)} - {max(sizes)} nodes",
                "dfs_scalability": "Good" if max(dfs_times) < 1.0 else "Moderate" if max(dfs_times) < 5.0 else "Poor",
                "pagerank_scalability": "Good" if max(pr_times) < 1.0 else "Moderate" if max(pr_times) < 5.0 else "Poor",
                "performance_ratios": {
                    "dfs_max_min_ratio": max(dfs_times) / min(dfs_times) if min(dfs_times) > 0 else float('inf'),
                    "pagerank_max_min_ratio": max(pr_times) / min(pr_times) if min(pr_times) > 0 else float('inf')
                }
            }
        
        return scalability
    
    def _summarize_correctness(self) -> Dict:
        """Summarize correctness test results."""
        if "standardized_tests" not in self.test_results:
            return {"error": "No correctness data available"}
        
        summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "algorithm_correctness": {}
        }
        
        for test_name, test_result in self.test_results["standardized_tests"].items():
            summary["total_tests"] += 1
            
            # Check DFS correctness
            dfs_tests = test_result.get("dfs_tests", {})
            dfs_passed = all([
                dfs_tests.get("traversal_consistency", {}).get("sets_equal", False),
                dfs_tests.get("connected_components", {}).get("coverage_complete", False),
                dfs_tests.get("connectivity", {}).get("consistent_with_components", False)
            ])
            
            # Check PageRank correctness
            pr_tests = test_result.get("pagerank_tests", {})
            pr_passed = all([
                pr_tests.get("basic_calculation", {}).get("scores_calculated", False),
                pr_tests.get("basic_calculation", {}).get("all_nodes_covered", False),
                pr_tests.get("basic_calculation", {}).get("scores_sum_normalized", False)
            ])
            
            if dfs_passed and pr_passed:
                summary["passed_tests"] += 1
        
        summary["pass_rate"] = summary["passed_tests"] / summary["total_tests"] if summary["total_tests"] > 0 else 0
        
        return summary
    
    def _generate_test_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check performance
        if self.performance_metrics:
            max_time = 0
            for results in self.performance_metrics.values():
                for alg_result in results.values():
                    if isinstance(alg_result, dict) and "avg_time" in alg_result:
                        max_time = max(max_time, alg_result["avg_time"])
            
            if max_time > 10:
                recommendations.append("Consider optimizing algorithms for better performance on large graphs")
        
        # Check correctness
        correctness = self._summarize_correctness()
        if correctness.get("pass_rate", 1.0) < 0.9:
            recommendations.append("Review algorithm implementations for correctness issues")
        
        # General recommendations
        recommendations.extend([
            "Consider implementing parallel versions for large-scale analysis",
            "Add more comprehensive error handling for edge cases",
            "Implement memory optimization for large graphs"
        ])
        
        return recommendations
    
    def _generate_performance_plots(self):
        """Generate performance visualization plots."""
        if not self.performance_metrics:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            sizes = []
            dfs_times = []
            pr_times = []
            
            for size_key, results in self.performance_metrics.items():
                size = int(size_key.split("_")[0])
                sizes.append(size)
                
                dfs_time = results.get("dfs", {}).get("recursive_avg_time", 0)
                pr_time = results.get("pagerank", {}).get("weighted_avg_time", 0)
                
                dfs_times.append(dfs_time)
                pr_times.append(pr_time)
            
            if sizes:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(sizes, dfs_times, 'bo-', label='DFS')
                plt.xlabel('Graph Size (nodes)')
                plt.ylabel('Average Time (seconds)')
                plt.title('DFS Performance Scaling')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(sizes, pr_times, 'ro-', label='PageRank')
                plt.xlabel('Graph Size (nodes)')
                plt.ylabel('Average Time (seconds)')
                plt.title('PageRank Performance Scaling')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(self.results_dir / "performance_plots.png")
                plt.close()
                
        except ImportError:
            pass  # matplotlib not available
    
    def _save_results(self, filename: str, data: Dict):
        """Save results to JSON file."""
        filepath = self.results_dir / filename
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_data = convert_numpy(data)
        
        with open(filepath, 'w') as f:
            json.dump(converted_data, f, indent=2, default=str)


# Convenience function for running complete test suite
def run_complete_test_suite(save_results: bool = True) -> Dict:
    """
    Run the complete test suite for all algorithms.
    
    Args:
        save_results: Whether to save results to files
    
    Returns:
        Dictionary with all test results
    """
    print("Starting complete test suite for Stock Market Tangle Project")
    print("=" * 60)
    
    tester = AlgorithmTester(save_results=save_results)
    
    # Run all test categories
    standardized_results = tester.run_standardized_test_cases()
    performance_results = tester.run_performance_benchmarks()
    correctness_results = tester.run_correctness_validation()
    
    # Generate comprehensive report
    comprehensive_report = tester.generate_comprehensive_report()
    
    print("\n" + "=" * 60)
    print("Test suite completed successfully!")
    print(f"Results saved to: {tester.results_dir}")
    
    return {
        "standardized_tests": standardized_results,
        "performance_benchmarks": performance_results,
        "correctness_validation": correctness_results,
        "comprehensive_report": comprehensive_report
    }