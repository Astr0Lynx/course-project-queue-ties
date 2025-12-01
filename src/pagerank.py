"""
PageRank Algorithm Implementation for Stock Market Tangle Project
Author: Khushi Dhingra
Description: Provides PageRank calculation for stock correlation networks.
             Identifies influential stocks in market correlation structure.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import time
from collections import defaultdict


class PageRank:
    """
    PageRank algorithm implementation for stock market influence analysis.
    
    Calculates importance scores for stocks based on their position
    in the correlation network structure.
    """
    
    def __init__(self, graph, damping_factor: float = 0.85):
        """
        Initialize PageRank with a graph.
        
        Args:
            graph: Graph object from graph.py
            damping_factor: Probability of following links (typically 0.85)
        """
        self.graph = graph
        self.damping_factor = damping_factor
        self.nodes = self.graph.get_nodes()
        self.num_nodes = len(self.nodes)
        self.node_to_index = {node: i for i, node in enumerate(self.nodes)}
        self.index_to_node = {i: node for i, node in enumerate(self.nodes)}
        
        # Initialize scores
        self.scores = {}
        self.iterations = 0
        self.convergence_history = []
    
    def calculate_pagerank(self, 
                          max_iterations: int = 100, 
                          tolerance: float = 1e-6,
                          weighted: bool = True) -> Dict[str, float]:
        """
        Calculate PageRank scores for all nodes.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence threshold
            weighted: Whether to use edge weights (correlations)
        
        Returns:
            Dictionary mapping nodes to PageRank scores
            
        Time Complexity: O(k * (V + E)) where k is iterations
        """
        if self.num_nodes == 0:
            return {}
        
        # Initialize scores uniformly
        initial_score = 1.0 / self.num_nodes
        current_scores = np.full(self.num_nodes, initial_score)
        new_scores = np.zeros(self.num_nodes)
        
        self.convergence_history = []
        
        for iteration in range(max_iterations):
            # Reset new scores
            new_scores.fill((1 - self.damping_factor) / self.num_nodes)
            
            # Calculate contributions from each node
            for i, node in enumerate(self.nodes):
                neighbors = self.graph.get_neighbors(node)
                
                if not neighbors:
                    # Dangling node - distribute score equally
                    contribution = self.damping_factor * current_scores[i] / self.num_nodes
                    new_scores += contribution
                else:
                    # Calculate outgoing weight sum for normalization
                    if weighted:
                        total_weight = sum(abs(weight) for weight in neighbors.values())
                    else:
                        total_weight = len(neighbors)
                    
                    # Distribute score to neighbors
                    for neighbor, weight in neighbors.items():
                        j = self.node_to_index[neighbor]
                        
                        if weighted and total_weight > 0:
                            # Use correlation strength as transition probability
                            transition_prob = abs(weight) / total_weight
                        else:
                            # Equal probability to all neighbors
                            transition_prob = 1.0 / len(neighbors)
                        
                        contribution = self.damping_factor * current_scores[i] * transition_prob
                        new_scores[j] += contribution
            
            # Check convergence
            diff = np.sum(np.abs(new_scores - current_scores))
            self.convergence_history.append(diff)
            
            if diff < tolerance:
                self.iterations = iteration + 1
                break
            
            # Update scores for next iteration
            current_scores, new_scores = new_scores, current_scores
        else:
            self.iterations = max_iterations
        
        # Convert to dictionary
        self.scores = {self.index_to_node[i]: score 
                      for i, score in enumerate(current_scores)}
        
        return self.scores.copy()
    
    def get_top_stocks(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N stocks by PageRank score.
        
        Args:
            n: Number of top stocks to return
        
        Returns:
            List of (stock, score) tuples sorted by score
        """
        if not self.scores:
            return []
        
        sorted_stocks = sorted(self.scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        return sorted_stocks[:n]
    
    def get_bottom_stocks(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get bottom N stocks by PageRank score.
        
        Args:
            n: Number of bottom stocks to return
        
        Returns:
            List of (stock, score) tuples sorted by score (ascending)
        """
        if not self.scores:
            return []
        
        sorted_stocks = sorted(self.scores.items(), 
                             key=lambda x: x[1])
        return sorted_stocks[:n]
    
    def analyze_convergence(self) -> Dict:
        """
        Analyze convergence behavior of PageRank algorithm.
        
        Returns:
            Dictionary with convergence statistics
        """
        if not self.convergence_history:
            return {'error': 'No convergence data available'}
        
        return {
            'iterations_to_convergence': self.iterations,
            'final_difference': self.convergence_history[-1] if self.convergence_history else 0,
            'convergence_history': self.convergence_history,
            'converged': self.iterations < len(self.convergence_history) + 1,
            'damping_factor': self.damping_factor
        }
    
    def compare_with_degree_centrality(self) -> Dict:
        """
        Compare PageRank scores with degree centrality.
        
        Returns:
            Dictionary with comparison metrics
        """
        if not self.scores:
            return {'error': 'PageRank scores not calculated'}
        
        # Calculate degree centrality
        degree_centrality = {}
        max_degree = max([self.graph.get_degree(node) for node in self.nodes]) if self.nodes else 1
        
        for node in self.nodes:
            degree_centrality[node] = self.graph.get_degree(node) / max_degree
        
        # Calculate correlation between PageRank and degree centrality
        pr_scores = [self.scores[node] for node in self.nodes]
        dc_scores = [degree_centrality[node] for node in self.nodes]
        
        if len(pr_scores) > 1:
            correlation = np.corrcoef(pr_scores, dc_scores)[0, 1]
        else:
            correlation = 0.0
        
        # Find stocks with high PageRank but low degree (hidden influencers)
        hidden_influencers = []
        for node in self.nodes:
            pr_rank = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
            dc_rank = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
            
            pr_position = next(i for i, (stock, _) in enumerate(pr_rank) if stock == node)
            dc_position = next(i for i, (stock, _) in enumerate(dc_rank) if stock == node)
            
            # High PageRank but low degree centrality
            if pr_position < len(self.nodes) // 4 and dc_position > len(self.nodes) // 2:
                hidden_influencers.append((node, self.scores[node], degree_centrality[node]))
        
        return {
            'correlation_with_degree': correlation,
            'degree_centrality': degree_centrality,
            'hidden_influencers': hidden_influencers,
            'top_pagerank': self.get_top_stocks(5),
            'top_degree': sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def sector_influence_analysis(self, stock_attributes: Dict) -> Dict:
        """
        Analyze PageRank influence by market sectors.
        
        Args:
            stock_attributes: Dictionary of stock attributes including sectors
        
        Returns:
            Dictionary with sector influence analysis
        """
        if not self.scores:
            return {'error': 'PageRank scores not calculated'}
        
        sector_scores = defaultdict(list)
        sector_stocks = defaultdict(list)
        
        # Group stocks by sector
        for stock, score in self.scores.items():
            if stock in stock_attributes:
                sector = stock_attributes[stock].get('sector', 'Unknown')
                sector_scores[sector].append(score)
                sector_stocks[sector].append((stock, score))
        
        # Calculate sector statistics
        sector_analysis = {}
        for sector, scores in sector_scores.items():
            sector_analysis[sector] = {
                'total_influence': sum(scores),
                'average_influence': np.mean(scores),
                'max_influence': max(scores),
                'min_influence': min(scores),
                'num_stocks': len(scores),
                'top_stocks': sorted(sector_stocks[sector], 
                                   key=lambda x: x[1], 
                                   reverse=True)[:3]
            }
        
        # Rank sectors by total influence
        sector_ranking = sorted(sector_analysis.items(), 
                              key=lambda x: x[1]['total_influence'], 
                              reverse=True)
        
        return {
            'sector_analysis': sector_analysis,
            'sector_ranking': sector_ranking,
            'most_influential_sector': sector_ranking[0] if sector_ranking else None,
            'sector_concentration': self._calculate_sector_concentration(sector_scores)
        }
    
    def _calculate_sector_concentration(self, sector_scores: Dict) -> float:
        """Calculate how concentrated influence is across sectors."""
        if not sector_scores:
            return 0.0
        
        total_scores = [sum(scores) for scores in sector_scores.values()]
        total_influence = sum(total_scores)
        
        if total_influence == 0:
            return 0.0
        
        # Calculate Herfindahl index for concentration
        sector_shares = [score / total_influence for score in total_scores]
        hhi = sum(share ** 2 for share in sector_shares)
        
        return hhi
    
    def simulate_node_removal(self, nodes_to_remove: List[str]) -> Dict:
        """
        Simulate the effect of removing specific nodes on PageRank scores.
        
        Args:
            nodes_to_remove: List of node names to remove
        
        Returns:
            Dictionary comparing original vs modified PageRank scores
        """
        # Store original scores
        original_scores = self.scores.copy()
        
        # Create modified graph (conceptually - we'll just modify calculation)
        remaining_nodes = [node for node in self.nodes if node not in nodes_to_remove]
        
        if not remaining_nodes:
            return {'error': 'Cannot remove all nodes'}
        
        # Temporarily modify the graph representation for calculation
        original_adjacency = {}
        for node in nodes_to_remove:
            if node in self.graph.adjacency_list:
                original_adjacency[node] = self.graph.adjacency_list[node].copy()
                # Remove the node
                del self.graph.adjacency_list[node]
                # Remove edges to this node from other nodes
                for other_node in self.graph.adjacency_list:
                    if node in self.graph.adjacency_list[other_node]:
                        del self.graph.adjacency_list[other_node][node]
        
        # Update node list and recalculate
        self.nodes = remaining_nodes
        self.num_nodes = len(self.nodes)
        self.node_to_index = {node: i for i, node in enumerate(self.nodes)}
        self.index_to_node = {i: node for i, node in enumerate(self.nodes)}
        
        # Calculate new PageRank scores
        modified_scores = self.calculate_pagerank()
        
        # Restore original graph
        for node, adjacency in original_adjacency.items():
            self.graph.adjacency_list[node] = adjacency
            for neighbor, weight in adjacency.items():
                if neighbor in self.graph.adjacency_list:
                    self.graph.adjacency_list[neighbor][node] = weight
        
        # Restore original node list
        self.nodes = self.graph.get_nodes()
        self.num_nodes = len(self.nodes)
        self.node_to_index = {node: i for i, node in enumerate(self.nodes)}
        self.index_to_node = {i: node for i, node in enumerate(self.nodes)}
        
        # Calculate score changes for remaining nodes
        score_changes = {}
        for node in remaining_nodes:
            original = original_scores.get(node, 0)
            modified = modified_scores.get(node, 0)
            score_changes[node] = modified - original
        
        # Find biggest gainers and losers
        gainers = sorted([(node, change) for node, change in score_changes.items() if change > 0], 
                        key=lambda x: x[1], reverse=True)[:5]
        losers = sorted([(node, change) for node, change in score_changes.items() if change < 0], 
                       key=lambda x: x[1])[:5]
        
        return {
            'nodes_removed': nodes_to_remove,
            'original_scores': {node: score for node, score in original_scores.items() 
                              if node in remaining_nodes},
            'modified_scores': modified_scores,
            'score_changes': score_changes,
            'biggest_gainers': gainers,
            'biggest_losers': losers,
            'total_score_redistribution': sum(abs(change) for change in score_changes.values())
        }
    
    def benchmark_performance(self, max_iterations: int = 100, num_runs: int = 5) -> Dict:
        """
        Benchmark PageRank algorithm performance.
        
        Args:
            max_iterations: Maximum iterations for PageRank
            num_runs: Number of runs for averaging
        
        Returns:
            Dictionary with performance metrics
        """
        weighted_times = []
        unweighted_times = []
        
        for _ in range(num_runs):
            # Benchmark weighted PageRank
            start_time = time.time()
            self.calculate_pagerank(max_iterations=max_iterations, weighted=True)
            end_time = time.time()
            weighted_times.append(end_time - start_time)
            
            # Benchmark unweighted PageRank
            start_time = time.time()
            self.calculate_pagerank(max_iterations=max_iterations, weighted=False)
            end_time = time.time()
            unweighted_times.append(end_time - start_time)
        
        return {
            'weighted_avg_time': np.mean(weighted_times),
            'unweighted_avg_time': np.mean(unweighted_times),
            'weighted_std_time': np.std(weighted_times),
            'unweighted_std_time': np.std(unweighted_times),
            'iterations_to_convergence': self.iterations,
            'graph_size': self.num_nodes,
            'num_edges': self.graph.num_edges,
            'num_runs': num_runs,
            'damping_factor': self.damping_factor
        }


def identify_market_influencers(graph, stock_attributes: Dict, top_n: int = 10) -> Dict:
    """
    Identify top market influencers using PageRank analysis.
    
    Args:
        graph: Stock correlation graph
        stock_attributes: Dictionary of stock attributes
        top_n: Number of top influencers to identify
    
    Returns:
        Dictionary with market influencer analysis
    """
    pagerank = PageRank(graph)
    
    # Calculate PageRank scores
    scores = pagerank.calculate_pagerank()
    
    # Get comprehensive analysis
    top_stocks = pagerank.get_top_stocks(top_n)
    sector_analysis = pagerank.sector_influence_analysis(stock_attributes)
    centrality_comparison = pagerank.compare_with_degree_centrality()
    convergence_info = pagerank.analyze_convergence()
    
    return {
        'top_influencers': top_stocks,
        'sector_influence': sector_analysis,
        'centrality_comparison': centrality_comparison,
        'convergence_analysis': convergence_info,
        'all_scores': scores
    }