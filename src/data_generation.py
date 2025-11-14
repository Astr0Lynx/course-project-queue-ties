"""
Data Generation Module for Stock Market Tangle Project
Author: Guntesh Singh
Description: Generates synthetic stock market data including returns, correlations,
             and stock attributes (volatility, stability) for graph-based analysis.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import json


class StockDataGenerator:
    """
    Generates synthetic stock market data for analysis.
    
    This class creates realistic stock return data, correlation matrices,
    and stock attributes that can be used to build stock correlation graphs.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
    
    def generate_stock_returns(self, 
                              num_stocks: int, 
                              num_days: int, 
                              mean_return: float = 0.001,
                              volatility_range: Tuple[float, float] = (0.01, 0.05)) -> pd.DataFrame:
        """
        Generate synthetic daily stock returns.
        
        Args:
            num_stocks: Number of stocks to generate
            num_days: Number of trading days
            mean_return: Average daily return
            volatility_range: Range of volatility (min, max)
        
        Returns:
            DataFrame with stock returns (rows=days, columns=stocks)
        """
        # Generate stock names
        stock_names = [f"STOCK_{i:03d}" for i in range(num_stocks)]
        
        # Assign random volatility to each stock
        volatilities = np.random.uniform(
            volatility_range[0], 
            volatility_range[1], 
            num_stocks
        )
        
        # Generate correlated returns using a factor model
        # This creates realistic correlation structure
        num_factors = max(3, num_stocks // 10)  # Number of market factors
        
        # Factor loadings (how much each stock is influenced by each factor)
        factor_loadings = np.random.randn(num_stocks, num_factors) * 0.5
        
        # Factor returns (market-wide influences)
        factor_returns = np.random.randn(num_days, num_factors) * 0.02
        
        # Idiosyncratic returns (stock-specific)
        idiosyncratic = np.random.randn(num_days, num_stocks)
        
        # Combine to get stock returns
        returns = (factor_returns @ factor_loadings.T + 
                  idiosyncratic * volatilities) + mean_return
        
        # Create DataFrame
        df_returns = pd.DataFrame(returns, columns=stock_names)
        
        return df_returns
    
    def calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix from stock returns.
        
        Args:
            returns: DataFrame of stock returns
        
        Returns:
            Correlation matrix as DataFrame
        """
        correlation_matrix = returns.corr()
        return correlation_matrix
    
    def generate_stock_attributes(self, 
                                  returns: pd.DataFrame,
                                  correlation_matrix: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate stock attributes based on returns and correlations.
        
        Args:
            returns: DataFrame of stock returns
            correlation_matrix: Correlation matrix
        
        Returns:
            Dictionary mapping stock names to their attributes
        """
        stock_attributes = {}
        
        for stock in returns.columns:
            # Calculate volatility (standard deviation of returns)
            volatility = returns[stock].std()
            
            # Calculate stability (inverse of volatility, normalized)
            stability = 1.0 / (1.0 + volatility)
            
            # Calculate mean return
            mean_return = returns[stock].mean()
            
            # Calculate average correlation with other stocks
            avg_correlation = correlation_matrix[stock].drop(stock).mean()
            
            # Classify stock based on volatility
            if volatility < 0.02:
                category = "stable"
            elif volatility < 0.035:
                category = "moderate"
            else:
                category = "volatile"
            
            stock_attributes[stock] = {
                'volatility': float(volatility),
                'stability': float(stability),
                'mean_return': float(mean_return),
                'avg_correlation': float(avg_correlation),
                'category': category
            }
        
        return stock_attributes
    
    def generate_dataset(self,
                        num_stocks: int,
                        num_days: int = 252,
                        scenario: str = "normal") -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Generate a complete dataset with returns, correlations, and attributes.
        
        Args:
            num_stocks: Number of stocks to generate
            num_days: Number of trading days (default 252 = 1 year)
            scenario: Market scenario ("normal", "stable", "volatile", "crash")
        
        Returns:
            Tuple of (returns DataFrame, correlation matrix, stock attributes dict)
        """
        # Adjust parameters based on scenario
        if scenario == "stable":
            mean_return = 0.0015
            volatility_range = (0.005, 0.02)
        elif scenario == "volatile":
            mean_return = 0.0005
            volatility_range = (0.03, 0.08)
        elif scenario == "crash":
            mean_return = -0.002
            volatility_range = (0.05, 0.15)
        else:  # normal
            mean_return = 0.001
            volatility_range = (0.01, 0.05)
        
        # Generate data
        returns = self.generate_stock_returns(
            num_stocks, 
            num_days, 
            mean_return, 
            volatility_range
        )
        
        correlation_matrix = self.calculate_correlation_matrix(returns)
        stock_attributes = self.generate_stock_attributes(returns, correlation_matrix)
        
        return returns, correlation_matrix, stock_attributes
    
    def save_dataset(self, 
                    returns: pd.DataFrame,
                    correlation_matrix: pd.DataFrame,
                    stock_attributes: Dict,
                    output_dir: str = "data",
                    prefix: str = "dataset") -> None:
        """
        Save generated dataset to files.
        
        Args:
            returns: DataFrame of stock returns
            correlation_matrix: Correlation matrix
            stock_attributes: Stock attributes dictionary
            output_dir: Directory to save files
            prefix: Prefix for output filenames
        """
        # Save returns
        returns.to_csv(f"{output_dir}/{prefix}_returns.csv", index=False)
        
        # Save correlation matrix
        correlation_matrix.to_csv(f"{output_dir}/{prefix}_correlations.csv")
        
        # Save attributes
        with open(f"{output_dir}/{prefix}_attributes.json", 'w') as f:
            json.dump(stock_attributes, f, indent=2)
        
        print(f"Dataset saved to {output_dir}/ with prefix '{prefix}'")
    
    def generate_multiple_scenarios(self, 
                                   num_stocks: int,
                                   output_dir: str = "data") -> None:
        """
        Generate datasets for multiple market scenarios.
        
        Args:
            num_stocks: Number of stocks
            output_dir: Directory to save files
        """
        scenarios = ["normal", "stable", "volatile", "crash"]
        
        for scenario in scenarios:
            print(f"\nGenerating {scenario} scenario...")
            returns, corr_matrix, attributes = self.generate_dataset(
                num_stocks, 
                scenario=scenario
            )
            
            self.save_dataset(
                returns, 
                corr_matrix, 
                attributes, 
                output_dir, 
                f"{scenario}_{num_stocks}"
            )


def main():
    """
    Example usage of the StockDataGenerator.
    """
    print("=== Stock Market Data Generator ===\n")
    
    generator = StockDataGenerator(seed=42)
    
    # Generate datasets of different sizes
    sizes = [50, 100, 200]
    
    for size in sizes:
        print(f"\n{'='*50}")
        print(f"Generating datasets with {size} stocks")
        print('='*50)
        generator.generate_multiple_scenarios(size, output_dir="data")
    
    print("\n" + "="*50)
    print("Data generation complete!")
    print("="*50)


if __name__ == "__main__":
    main()
