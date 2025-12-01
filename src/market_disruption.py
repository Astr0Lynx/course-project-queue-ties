"""
Market Disruption Simulations for Stock Market Tangle Project
Author: Khushi Dhingra
Description: Simulates various market disruption scenarios and analyzes their
             impact on network structure using DFS and PageRank algorithms.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import copy
from collections import defaultdict
import random

from .dfs import DFS, analyze_market_connectivity
from .pagerank import PageRank, identify_market_influencers

# Import additional algorithms for comprehensive disruption analysis
try:
    from .union_find import UnionFind
except ImportError:
    UnionFind = None

try:
    from .bfs import BFS
except ImportError:
    BFS = None

try:
    from .louvain import detect_communities as louvain_detect
except ImportError:
    louvain_detect = None

try:
    from .girvan_newman import GirvanNewman
except ImportError:
    GirvanNewman = None

try:
    from .node2vec import Node2Vec
except ImportError:
    Node2Vec = None


class MarketDisruptionSimulator:
    """
    Simulates various market disruption scenarios and analyzes their impact
    on stock correlation network structure and influence patterns.
    """
    
    def __init__(self, graph, stock_attributes: Dict):
        """
        Initialize simulator with a stock correlation graph.
        
        Args:
            graph: Stock correlation graph from graph.py
            stock_attributes: Dictionary of stock attributes including sectors
        """
        self.original_graph = graph
        self.stock_attributes = stock_attributes
        self.simulation_results = {}
        
    def simulate_market_crash(self, severity: float = 0.7, 
                            correlation_threshold: float = 0.3) -> Dict:
        """
        Simulate a market crash by increasing correlations and analyzing impact.
        
        During market crashes, stocks tend to become more correlated as
        panic selling affects all assets similarly.
        
        Args:
            severity: How much correlations increase (0.0 to 1.0)
            correlation_threshold: New threshold for edge creation
        
        Returns:
            Dictionary with crash impact analysis
        """
        print(f"Simulating market crash (severity: {severity})")
        
        # Create a copy of the graph for simulation
        crash_graph = self._copy_graph()
        
        # Modify correlations to simulate crash conditions
        edges_to_add = []
        edges_to_modify = []
        
        nodes = crash_graph.get_nodes()
        
        # Increase existing correlations
        for node1, node2, weight in crash_graph.get_edges():
            new_weight = min(1.0, weight + (severity * (1.0 - weight)))
            crash_graph.adjacency_list[node1][node2] = new_weight
            crash_graph.adjacency_list[node2][node1] = new_weight
            edges_to_modify.append((node1, node2, weight, new_weight))
        
        # Add new edges between previously uncorrelated stocks
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if not crash_graph.has_edge(node1, node2):
                    # Create new correlation based on crash severity
                    new_correlation = correlation_threshold + (severity * 0.3)
                    if new_correlation >= correlation_threshold:
                        crash_graph.add_edge(node1, node2, new_correlation)
                        edges_to_add.append((node1, node2, new_correlation))
        
        # Analyze connectivity changes
        original_connectivity = self._analyze_connectivity(self.original_graph)
        crash_connectivity = self._analyze_connectivity(crash_graph)
        
        # Analyze influence changes
        original_influence = self._analyze_influence(self.original_graph)
        crash_influence = self._analyze_influence(crash_graph)
        
        crash_results = {
            'scenario': 'Market Crash',
            'severity': severity,
            'correlation_threshold': correlation_threshold,
            'edges_added': len(edges_to_add),
            'edges_modified': len(edges_to_modify),
            'connectivity_changes': self._compare_connectivity(original_connectivity, crash_connectivity),
            'influence_changes': self._compare_influence(original_influence, crash_influence),
            'sector_impact': self._analyze_sector_impact(crash_graph, original_influence, crash_influence),
            'network_fragility': self._calculate_network_fragility(crash_graph)
        }
        
        self.simulation_results['market_crash'] = crash_results
        return crash_results
    
    def simulate_sector_collapse(self, target_sector: str, 
                               impact_radius: int = 2) -> Dict:
        """
        Simulate the collapse of a specific market sector.
        
        Args:
            target_sector: Sector to simulate collapse for
            impact_radius: How many network hops the impact spreads
        
        Returns:
            Dictionary with sector collapse analysis
        """
        print(f"Simulating {target_sector} sector collapse")
        
        # Identify stocks in the target sector
        sector_stocks = [stock for stock, attrs in self.stock_attributes.items()
                        if attrs.get('sector') == target_sector and stock in self.original_graph.get_nodes()]
        
        if not sector_stocks:
            return {'error': f'No stocks found for sector: {target_sector}'}
        
        # Create graph without sector stocks
        collapsed_graph = self._copy_graph()
        
        # Remove sector stocks and their connections
        removed_connections = []
        for stock in sector_stocks:
            if stock in collapsed_graph.adjacency_list:
                neighbors = list(collapsed_graph.get_neighbors(stock).keys())
                for neighbor in neighbors:
                    weight = collapsed_graph.get_edge_weight(stock, neighbor)
                    removed_connections.append((stock, neighbor, weight))
                    collapsed_graph.remove_edge(stock, neighbor)
                
                del collapsed_graph.adjacency_list[stock]
                collapsed_graph.num_nodes -= 1
        
        # Analyze ripple effects on remaining network
        original_connectivity = self._analyze_connectivity(self.original_graph)
        collapsed_connectivity = self._analyze_connectivity(collapsed_graph)
        
        original_influence = self._analyze_influence(self.original_graph)
        collapsed_influence = self._analyze_influence(collapsed_graph)
        
        # Identify most affected remaining stocks
        influence_changes = {}
        for stock in collapsed_graph.get_nodes():
            if stock in original_influence['all_scores'] and stock in collapsed_influence['all_scores']:
                original_score = original_influence['all_scores'][stock]
                new_score = collapsed_influence['all_scores'][stock]
                influence_changes[stock] = new_score - original_score
        
        # Find biggest gainers and losers
        gainers = sorted([(stock, change) for stock, change in influence_changes.items() if change > 0],
                        key=lambda x: x[1], reverse=True)[:10]
        losers = sorted([(stock, change) for stock, change in influence_changes.items() if change < 0],
                       key=lambda x: x[1])[:10]
        
        collapse_results = {
            'scenario': 'Sector Collapse',
            'target_sector': target_sector,
            'stocks_removed': sector_stocks,
            'connections_lost': len(removed_connections),
            'connectivity_changes': self._compare_connectivity(original_connectivity, collapsed_connectivity),
            'influence_changes': self._compare_influence(original_influence, collapsed_influence),
            'biggest_gainers': gainers,
            'biggest_losers': losers,
            'network_fragmentation': self._calculate_fragmentation(collapsed_connectivity),
            'systemic_risk_score': self._calculate_systemic_risk(collapsed_connectivity, original_connectivity)
        }
        
        self.simulation_results['sector_collapse'] = collapse_results
        return collapse_results
    
    def simulate_key_player_removal(self, removal_strategy: str = 'top_pagerank', 
                                   num_players: int = 5) -> Dict:
        """
        Simulate removal of key market players based on different strategies.
        
        Args:
            removal_strategy: 'top_pagerank', 'highest_degree', 'random', or 'sector_leaders'
            num_players: Number of players to remove
        
        Returns:
            Dictionary with key player removal analysis
        """
        print(f"Simulating removal of {num_players} key players using {removal_strategy} strategy")
        
        # Identify players to remove based on strategy
        if removal_strategy == 'top_pagerank':
            influence_analysis = self._analyze_influence(self.original_graph)
            top_stocks = influence_analysis['top_influencers'][:num_players]
            players_to_remove = [stock for stock, _ in top_stocks]
            
        elif removal_strategy == 'highest_degree':
            node_degrees = [(node, self.original_graph.get_degree(node)) 
                           for node in self.original_graph.get_nodes()]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            players_to_remove = [stock for stock, _ in node_degrees[:num_players]]
            
        elif removal_strategy == 'sector_leaders':
            players_to_remove = self._identify_sector_leaders(num_players)
            
        elif removal_strategy == 'random':
            all_nodes = self.original_graph.get_nodes()
            players_to_remove = random.sample(all_nodes, min(num_players, len(all_nodes)))
            
        else:
            return {'error': f'Unknown removal strategy: {removal_strategy}'}
        
        # Create graph without key players
        reduced_graph = self._copy_graph()
        removed_connections = []
        
        for player in players_to_remove:
            if player in reduced_graph.adjacency_list:
                neighbors = list(reduced_graph.get_neighbors(player).keys())
                for neighbor in neighbors:
                    weight = reduced_graph.get_edge_weight(player, neighbor)
                    removed_connections.append((player, neighbor, weight))
                    reduced_graph.remove_edge(player, neighbor)
                
                del reduced_graph.adjacency_list[player]
                reduced_graph.num_nodes -= 1
        
        # Analyze impact
        original_connectivity = self._analyze_connectivity(self.original_graph)
        reduced_connectivity = self._analyze_connectivity(reduced_graph)
        
        original_influence = self._analyze_influence(self.original_graph)
        reduced_influence = self._analyze_influence(reduced_graph)
        
        # Calculate network resilience metrics
        resilience_score = self._calculate_network_resilience(
            original_connectivity, reduced_connectivity, len(players_to_remove)
        )
        
        removal_results = {
            'scenario': 'Key Player Removal',
            'removal_strategy': removal_strategy,
            'players_removed': players_to_remove,
            'connections_lost': len(removed_connections),
            'connectivity_changes': self._compare_connectivity(original_connectivity, reduced_connectivity),
            'influence_changes': self._compare_influence(original_influence, reduced_influence),
            'network_resilience_score': resilience_score,
            'cascade_effects': self._analyze_cascade_effects(reduced_graph, players_to_remove)
        }
        
        self.simulation_results['key_player_removal'] = removal_results
        return removal_results
    
    def simulate_correlation_threshold_changes(self, 
                                             thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) -> Dict:
        """
        Simulate the effect of changing correlation thresholds on network structure.
        
        Args:
            thresholds: List of correlation thresholds to test
        
        Returns:
            Dictionary with threshold sensitivity analysis
        """
        print("Simulating correlation threshold sensitivity")
        
        from .graph import build_graph_from_correlation
        from .data_generation import StockDataGenerator
        
        # We need the original correlation matrix - let's regenerate it
        generator = StockDataGenerator(seed=42)
        num_stocks = len(self.original_graph.get_nodes())
        returns, corr_matrix, stock_attrs = generator.generate_dataset(num_stocks, scenario="normal")
        
        threshold_results = {}
        
        for threshold in thresholds:
            # Build graph with this threshold
            thresh_graph = build_graph_from_correlation(corr_matrix, stock_attrs, threshold)
            
            # Analyze this configuration
            connectivity = self._analyze_connectivity(thresh_graph)
            influence = self._analyze_influence(thresh_graph)
            
            threshold_results[threshold] = {
                'num_nodes': thresh_graph.num_nodes,
                'num_edges': thresh_graph.num_edges,
                'connectivity_ratio': connectivity['connectivity_ratio'],
                'num_components': connectivity['num_components'],
                'largest_component_size': connectivity['largest_component_size'],
                'top_influencer_score': influence['top_influencers'][0][1] if influence['top_influencers'] else 0,
                'influence_concentration': self._calculate_influence_concentration(influence['all_scores'])
            }
        
        # Analyze trends
        sensitivity_analysis = {
            'scenario': 'Threshold Sensitivity',
            'thresholds_tested': thresholds,
            'threshold_results': threshold_results,
            'sensitivity_metrics': self._calculate_threshold_sensitivity(threshold_results),
            'optimal_threshold': self._find_optimal_threshold(threshold_results)
        }
        
        self.simulation_results['threshold_sensitivity'] = sensitivity_analysis
        return sensitivity_analysis
    
    def run_comprehensive_stress_test(self) -> Dict:
        """
        Run comprehensive stress test with multiple disruption scenarios.
        
        Returns:
            Dictionary with comprehensive stress test results
        """
        print("Running comprehensive market stress test...")
        
        stress_results = {}
        
        # Test 1: Market Crash Scenarios
        crash_severities = [0.3, 0.5, 0.7, 0.9]
        stress_results['crash_tests'] = {}
        for severity in crash_severities:
            stress_results['crash_tests'][f'severity_{severity}'] = self.simulate_market_crash(severity)
        
        # Test 2: Multiple Sector Collapses
        sectors = set(attrs.get('sector', 'Unknown') for attrs in self.stock_attributes.values())
        stress_results['sector_tests'] = {}
        for sector in list(sectors)[:3]:  # Test top 3 sectors
            if sector != 'Unknown':
                stress_results['sector_tests'][sector] = self.simulate_sector_collapse(sector)
        
        # Test 3: Key Player Removal Strategies
        strategies = ['top_pagerank', 'highest_degree', 'random']
        stress_results['key_player_tests'] = {}
        for strategy in strategies:
            stress_results['key_player_tests'][strategy] = self.simulate_key_player_removal(strategy)
        
        # Test 4: Threshold Sensitivity
        stress_results['threshold_test'] = self.simulate_correlation_threshold_changes()
        
        # Generate overall resilience report
        stress_results['overall_assessment'] = self._generate_resilience_report(stress_results)
        
        self.simulation_results['comprehensive_stress_test'] = stress_results
        return stress_results
    
    # Helper methods
    def _copy_graph(self):
        """Create a deep copy of the graph for simulation."""
        new_graph = type(self.original_graph)()
        
        # Copy nodes and attributes
        for node in self.original_graph.get_nodes():
            attrs = self.original_graph.node_attributes.get(node, {})
            new_graph.add_node(node, attrs)
        
        # Copy edges
        for node1, node2, weight in self.original_graph.get_edges():
            new_graph.add_edge(node1, node2, weight)
        
        return new_graph
    
    def _analyze_connectivity(self, graph):
        """Analyze connectivity using DFS, Union-Find, and BFS for comprehensive analysis."""
        # Base connectivity from DFS
        connectivity = analyze_market_connectivity(graph, self.stock_attributes)
        
        # Add Union-Find component analysis if available
        if UnionFind is not None:
            nodes = list(graph.get_nodes())
            uf = UnionFind(len(nodes))
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            
            for u in nodes:
                for v in graph.get_neighbors(u):
                    if v in node_to_idx:
                        uf.union(node_to_idx[u], node_to_idx[v])
            
            connectivity['union_find_components'] = len(set(uf.find(i) for i in range(len(nodes))))
        
        # Add BFS path analysis if available
        if BFS is not None:
            bfs = BFS(graph)
            if nodes:
                # Analyze average path lengths from random samples
                sample_size = min(5, len(nodes))
                import random
                sample_nodes = random.sample(nodes, sample_size)
                avg_paths = []
                for node in sample_nodes:
                    distances = bfs.shortest_paths(node)
                    finite_distances = [d for d in distances.values() if d != float('inf')]
                    if finite_distances:
                        avg_paths.append(sum(finite_distances) / len(finite_distances))
                
                connectivity['avg_path_length'] = sum(avg_paths) / len(avg_paths) if avg_paths else float('inf')
        
        return connectivity
    
    def _analyze_influence(self, graph):
        """Analyze influence using PageRank and community detection algorithms."""
        # Base influence from PageRank
        influence = identify_market_influencers(graph, self.stock_attributes)
        
        # Add Louvain community detection if available
        if louvain_detect is not None:
            try:
                communities = louvain_detect(graph)
                influence['louvain_communities'] = {
                    'num_communities': len(set(communities.values())),
                    'modularity': self._calculate_modularity(graph, communities)
                }
            except Exception:
                pass
        
        # Add Girvan-Newman community detection if available
        if GirvanNewman is not None:
            try:
                gn = GirvanNewman(graph)
                gn_communities = gn.detect_communities(max_communities=10)
                influence['girvan_newman_communities'] = len(gn_communities)
            except Exception:
                pass
        
        # Add Node2Vec embeddings if available
        if Node2Vec is not None:
            try:
                n2v = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200)
                embeddings = n2v.fit()
                influence['node2vec_embedding_dim'] = len(next(iter(embeddings.values()))) if embeddings else 0
            except Exception:
                pass
        
        return influence
    
    def _compare_connectivity(self, original, modified):
        """Compare connectivity metrics between two graphs."""
        return {
            'component_change': modified['num_components'] - original['num_components'],
            'connectivity_ratio_change': modified['connectivity_ratio'] - original['connectivity_ratio'],
            'largest_component_change': modified['largest_component_size'] - original['largest_component_size'],
            'fragmentation_increase': (modified['num_components'] - original['num_components']) / original['total_nodes']
        }
    
    def _compare_influence(self, original, modified):
        """Compare influence metrics between two graphs."""
        # Compare top influencers
        orig_top = set(stock for stock, _ in original['top_influencers'][:5])
        mod_top = set(stock for stock, _ in modified['top_influencers'][:5])
        
        return {
            'top_influencer_stability': len(orig_top & mod_top) / len(orig_top) if orig_top else 0,
            'influence_redistribution': self._calculate_influence_redistribution(
                original['all_scores'], modified['all_scores']
            )
        }
    
    def _calculate_modularity(self, graph, communities):
        """Calculate modularity score for community structure."""
        nodes = list(graph.get_nodes())
        m = sum(len(list(graph.get_neighbors(node))) for node in nodes) / 2  # Total edges
        
        if m == 0:
            return 0.0
        
        modularity = 0.0
        for node_i in nodes:
            for node_j in nodes:
                if communities.get(node_i) == communities.get(node_j):
                    # Check if edge exists
                    a_ij = 1 if node_j in graph.get_neighbors(node_i) else 0
                    k_i = len(list(graph.get_neighbors(node_i)))
                    k_j = len(list(graph.get_neighbors(node_j)))
                    modularity += a_ij - (k_i * k_j) / (2 * m)
        
        return modularity / (2 * m)
    
    def _analyze_sector_impact(self, graph, original_influence, modified_influence):
        """Analyze impact on different market sectors."""
        sector_impact = defaultdict(lambda: {'score_change': 0, 'stocks_affected': 0})
        
        for stock in graph.get_nodes():
            if stock in self.stock_attributes:
                sector = self.stock_attributes[stock].get('sector', 'Unknown')
                if stock in original_influence['all_scores'] and stock in modified_influence['all_scores']:
                    change = modified_influence['all_scores'][stock] - original_influence['all_scores'][stock]
                    sector_impact[sector]['score_change'] += change
                    sector_impact[sector]['stocks_affected'] += 1
        
        return dict(sector_impact)
    
    def _calculate_network_fragility(self, graph):
        """Calculate network fragility score."""
        dfs = DFS(graph)
        connectivity_info = dfs.get_connectivity_info()
        
        if connectivity_info['total_nodes'] == 0:
            return 1.0
        
        # Fragility based on number of components and component size distribution
        component_sizes = connectivity_info['component_sizes']
        if not component_sizes:
            return 1.0
        
        # Normalized Gini coefficient for component size inequality
        sorted_sizes = sorted(component_sizes)
        n = len(sorted_sizes)
        cumsum = np.cumsum(sorted_sizes)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_sizes) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
        
        return gini
    
    def _calculate_fragmentation(self, connectivity_info):
        """Calculate network fragmentation score."""
        if connectivity_info['total_nodes'] == 0:
            return 1.0
        
        return 1 - (connectivity_info['largest_component_size'] / connectivity_info['total_nodes'])
    
    def _calculate_systemic_risk(self, modified_connectivity, original_connectivity):
        """Calculate systemic risk score based on connectivity changes."""
        original_ratio = original_connectivity['connectivity_ratio']
        modified_ratio = modified_connectivity['connectivity_ratio']
        
        if original_ratio == 0:
            return 1.0
        
        return 1 - (modified_ratio / original_ratio)
    
    def _identify_sector_leaders(self, num_leaders):
        """Identify sector leaders for removal."""
        influence_analysis = self._analyze_influence(self.original_graph)
        sector_leaders = {}
        
        # Find top stock in each sector
        for stock, score in influence_analysis['top_influencers']:
            if stock in self.stock_attributes:
                sector = self.stock_attributes[stock].get('sector', 'Unknown')
                if sector not in sector_leaders or score > sector_leaders[sector][1]:
                    sector_leaders[sector] = (stock, score)
        
        # Return top leaders across sectors
        leaders = [stock for stock, score in sector_leaders.values()]
        return leaders[:num_leaders]
    
    def _calculate_network_resilience(self, original, modified, num_removed):
        """Calculate network resilience score."""
        if original['total_nodes'] == 0:
            return 0.0
        
        removal_fraction = num_removed / original['total_nodes']
        connectivity_retained = modified['connectivity_ratio'] / original['connectivity_ratio'] if original['connectivity_ratio'] > 0 else 0
        
        # Resilience is connectivity retained relative to fraction removed
        expected_degradation = removal_fraction
        actual_degradation = 1 - connectivity_retained
        
        if expected_degradation == 0:
            return 1.0
        
        return max(0, 1 - (actual_degradation / expected_degradation))
    
    def _analyze_cascade_effects(self, graph, removed_nodes):
        """Analyze cascade effects of node removal."""
        # This is a simplified cascade analysis
        remaining_nodes = graph.get_nodes()
        cascade_score = 0
        
        for node in remaining_nodes:
            # Check if this node lost significant connections
            original_degree = self.original_graph.get_degree(node)
            current_degree = graph.get_degree(node)
            
            if original_degree > 0:
                degree_loss = (original_degree - current_degree) / original_degree
                cascade_score += degree_loss
        
        return cascade_score / len(remaining_nodes) if remaining_nodes else 0
    
    def _calculate_influence_concentration(self, scores):
        """Calculate influence concentration using Herfindahl index."""
        if not scores:
            return 0
        
        total = sum(scores.values())
        if total == 0:
            return 0
        
        shares = [score / total for score in scores.values()]
        return sum(share ** 2 for share in shares)
    
    def _calculate_influence_redistribution(self, original_scores, modified_scores):
        """Calculate how much influence was redistributed."""
        total_change = 0
        common_stocks = set(original_scores.keys()) & set(modified_scores.keys())
        
        for stock in common_stocks:
            change = abs(modified_scores[stock] - original_scores[stock])
            total_change += change
        
        return total_change / len(common_stocks) if common_stocks else 0
    
    def _calculate_threshold_sensitivity(self, threshold_results):
        """Calculate sensitivity metrics for threshold analysis."""
        thresholds = sorted(threshold_results.keys())
        
        # Calculate derivatives (rate of change)
        edge_changes = []
        connectivity_changes = []
        
        for i in range(1, len(thresholds)):
            prev_thresh = thresholds[i-1]
            curr_thresh = thresholds[i]
            
            edge_change = (threshold_results[curr_thresh]['num_edges'] - 
                          threshold_results[prev_thresh]['num_edges']) / (curr_thresh - prev_thresh)
            edge_changes.append(edge_change)
            
            conn_change = (threshold_results[curr_thresh]['connectivity_ratio'] - 
                          threshold_results[prev_thresh]['connectivity_ratio']) / (curr_thresh - prev_thresh)
            connectivity_changes.append(conn_change)
        
        return {
            'edge_sensitivity': np.mean(np.abs(edge_changes)) if edge_changes else 0,
            'connectivity_sensitivity': np.mean(np.abs(connectivity_changes)) if connectivity_changes else 0,
            'most_sensitive_range': self._find_most_sensitive_range(thresholds, edge_changes)
        }
    
    def _find_most_sensitive_range(self, thresholds, changes):
        """Find the threshold range with highest sensitivity."""
        if not changes:
            return None
        
        max_change_idx = np.argmax(np.abs(changes))
        return (thresholds[max_change_idx], thresholds[max_change_idx + 1])
    
    def _find_optimal_threshold(self, threshold_results):
        """Find optimal threshold balancing connectivity and sparsity."""
        scores = {}
        
        for threshold, results in threshold_results.items():
            # Score based on connectivity ratio and edge density
            connectivity_score = results['connectivity_ratio']
            
            # Penalize too many or too few edges
            edge_ratio = results['num_edges'] / (results['num_nodes'] * (results['num_nodes'] - 1) / 2) if results['num_nodes'] > 1 else 0
            edge_score = 1 - abs(edge_ratio - 0.1)  # Target ~10% density
            
            scores[threshold] = 0.7 * connectivity_score + 0.3 * edge_score
        
        return max(scores.items(), key=lambda x: x[1]) if scores else None
    
    def _generate_resilience_report(self, stress_results):
        """Generate overall resilience assessment."""
        resilience_scores = []
        
        # Collect resilience indicators from different tests
        for test_type, results in stress_results.items():
            if test_type == 'crash_tests':
                for severity, crash_result in results.items():
                    fragility = crash_result.get('network_fragility', 1.0)
                    resilience_scores.append(1 - fragility)
            
            elif test_type == 'key_player_tests':
                for strategy, removal_result in results.items():
                    resilience = removal_result.get('network_resilience_score', 0)
                    resilience_scores.append(resilience)
        
        overall_resilience = np.mean(resilience_scores) if resilience_scores else 0
        
        # Classify resilience level
        if overall_resilience > 0.8:
            resilience_level = "High"
        elif overall_resilience > 0.6:
            resilience_level = "Moderate"
        elif overall_resilience > 0.4:
            resilience_level = "Low"
        else:
            resilience_level = "Very Low"
        
        return {
            'overall_resilience_score': overall_resilience,
            'resilience_level': resilience_level,
            'individual_scores': resilience_scores,
            'main_vulnerabilities': self._identify_main_vulnerabilities(stress_results),
            'recommendations': self._generate_recommendations(overall_resilience, stress_results)
        }
    
    def _identify_main_vulnerabilities(self, stress_results):
        """Identify main vulnerabilities from stress test results."""
        vulnerabilities = []
        
        # Check crash vulnerability
        if 'crash_tests' in stress_results:
            high_severity_result = stress_results['crash_tests'].get('severity_0.7')
            if high_severity_result and high_severity_result.get('network_fragility', 0) > 0.5:
                vulnerabilities.append("High vulnerability to market crashes")
        
        # Check sector concentration
        if 'sector_tests' in stress_results:
            for sector, result in stress_results['sector_tests'].items():
                systemic_risk = result.get('systemic_risk_score', 0)
                if systemic_risk > 0.3:
                    vulnerabilities.append(f"High dependence on {sector} sector")
        
        # Check key player dependence
        if 'key_player_tests' in stress_results:
            pagerank_result = stress_results['key_player_tests'].get('top_pagerank')
            if pagerank_result and pagerank_result.get('network_resilience_score', 1) < 0.5:
                vulnerabilities.append("High dependence on key market players")
        
        return vulnerabilities
    
    def _generate_recommendations(self, resilience_score, stress_results):
        """Generate recommendations based on stress test results."""
        recommendations = []
        
        if resilience_score < 0.6:
            recommendations.append("Consider diversifying portfolio across more sectors")
            recommendations.append("Monitor key influencer stocks for early warning signals")
        
        if resilience_score < 0.4:
            recommendations.append("Implement stress testing protocols")
            recommendations.append("Consider reducing exposure to highly correlated assets")
        
        # Specific recommendations based on vulnerabilities
        vulnerabilities = self._identify_main_vulnerabilities(stress_results)
        if "High vulnerability to market crashes" in vulnerabilities:
            recommendations.append("Implement crash protection mechanisms")
        
        if any("sector" in vuln for vuln in vulnerabilities):
            recommendations.append("Increase cross-sector diversification")
        
        return recommendations

# Provide a stable function name expected by other modules (benchmarks.py)
def girvan_newman_algorithm(graph, max_iterations: int = 5):
    """
    Wrapper for Girvan-Newman community detection used by benchmarks.
    Uses the GirvanNewman class if available; if not available or on error,
    returns a safe fallback (all nodes assigned to a single community).
    """
    if GirvanNewman is None:
        raise ImportError("GirvanNewman implementation is not available in this environment")
    try:
        gn = GirvanNewman(graph)
        # Try common parameter names used by different implementations
        try:
            communities = gn.detect_communities(max_iterations=max_iterations)
        except TypeError:
            communities = gn.detect_communities(max_communities=max_iterations)
        return communities
    except Exception:
        # Fallback: assign every node to a single community (no split)
        try:
            return {node: 0 for node in graph.get_nodes()}
        except Exception:
            return {}