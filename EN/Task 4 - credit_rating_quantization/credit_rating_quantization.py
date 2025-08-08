"""
Credit Rating Quantization Module

A comprehensive module for mapping FICO scores to discrete credit ratings using
advanced quantization techniques. This module provides optimal bucketing strategies
to categorize continuous credit scores into meaningful risk segments.

Purpose: Transforms continuous FICO scores (300-850) into discrete credit rating
         categories for risk assessment and decision-making processes. Lower rating
         numbers indicate better creditworthiness (e.g., 1=Prime, 10=Subprime).

Author: Curtis Yan
Date: August 7, 2025

Key Optimizations:
- Precomputed cumulative statistics for efficient MSE calculations
- Optimized dynamic programming implementation with reduced complexity
- Memory-efficient array operations and minimal allocations
- Optional Numba JIT compilation for performance-critical sections
"""

import platform
import warnings
from typing import List, Tuple, Callable, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Try to import numba for JIT compilation (robust fallback)
try:
    from numba import njit, prange  # prange for potential parallel loops
    HAS_NUMBA = True
except ImportError:
    # Fallback for when numba is not available
    def njit(*args, **kwargs):
        # allow usage as decorator with or without params
        if args and callable(args[0]) and not kwargs:
            return args[0]
        def wrapper(func):
            return func
        return wrapper
    def prange(*args, **kwargs):  # type: ignore
        return range(*args)
    HAS_NUMBA = False

# Matplotlib font settings by platform (to properly render minus signs and CJK if present)
if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS"]
elif platform.system() == "Windows":
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class CreditRatingQuantizer:
    """
    Advanced credit rating quantization framework for mapping FICO scores to discrete risk categories.
    
    This quantizer implements state-of-the-art optimization algorithms to segment continuous 
    credit scores into meaningful risk buckets, facilitating more accurate risk assessment and 
    capital allocation decisions.
    
    Supported optimization methods:
    1. MSE (Mean Squared Error): Minimizes information loss by reducing squared deviations 
       between original scores and their quantized representations
    2. Log-likelihood: Maximizes the statistical likelihood function considering observed 
       default probabilities, providing risk-sensitive bucketing
    
    Attributes:
        method (str): Optimization criterion ('mse' or 'log_likelihood')
        boundaries (List[float]): Optimal bucket boundaries after fitting
        rating_map (Callable): Function mapping FICO scores to rating categories
        num_buckets (int): Number of discrete rating categories
    """
    
    def __init__(self, method='mse'):
        """
        Initialize the credit rating quantizer with specified optimization criterion.
        
        Parameters:
            method (str): Optimization method selection
                - 'mse': Mean Squared Error minimization for balanced bucketing
                - 'log_likelihood': Maximum likelihood estimation for risk-based segmentation
        """
        self.method = method
        self.boundaries = None
        self.rating_map = None
        self.num_buckets = None
        
    def fit(self, fico_scores: np.ndarray, defaults: np.ndarray = None, 
            num_buckets: int = 10) -> 'CreditRatingQuantizer':
        """
        Fit the quantizer to historical credit score data using dynamic programming optimization.
        
        This method identifies optimal bucket boundaries that minimize the chosen objective 
        function while maintaining monotonic risk ordering.
        
        Parameters:
            fico_scores (np.ndarray): Array of FICO credit scores (valid range: 300-850)
            defaults (np.ndarray, optional): Binary default indicators (0=performing, 1=defaulted)
                Required for log-likelihood optimization method
            num_buckets (int): Target number of rating categories (e.g., 10 for decile ratings)
            
        Returns:
            CreditRatingQuantizer: Fitted quantizer instance for method chaining
            
        Raises:
            ValueError: If defaults array is not provided for log-likelihood method
        """
        self.num_buckets = num_buckets
        
        # Sort scores for easier processing
        sorted_indices = np.argsort(fico_scores)
        sorted_scores = fico_scores[sorted_indices]
        
        if self.method == 'mse':
            self.boundaries = self._optimize_mse_fast(sorted_scores, num_buckets)
        elif self.method == 'log_likelihood':
            if defaults is None:
                raise ValueError("Default data required for log-likelihood method")
            sorted_defaults = defaults[sorted_indices]
            self.boundaries = self._optimize_log_likelihood_fast(sorted_scores, 
                                                                sorted_defaults, 
                                                                num_buckets)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        # Create rating mapping function
        self.rating_map = self._create_rating_function()
        
        return self
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _precompute_mse_stats(sorted_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precompute cumulative sums and squared sums for fast MSE calculations.
        Note: Implemented as a static function to be compatible with Numba (no `self`).
        """
        n = len(sorted_scores)
        cum_sum = np.zeros(n + 1)
        cum_sum2 = np.zeros(n + 1)
        cum_sum[1:] = np.cumsum(sorted_scores)
        cum_sum2[1:] = np.cumsum(sorted_scores ** 2)
        return cum_sum, cum_sum2
    
    def _compute_bucket_mse(self, cum_sum: np.ndarray, cum_sum2: np.ndarray, 
                           start: int, end: int) -> float:
        """
        Compute MSE for a bucket using precomputed cumulative statistics.
        
        MSE = E[X²] - (E[X])² = sum(x²)/n - (sum(x)/n)²
        
        Parameters:
            cum_sum (np.ndarray): Cumulative sum array
            cum_sum2 (np.ndarray): Cumulative sum of squares array
            start (int): Start index (inclusive)
            end (int): End index (exclusive)
            
        Returns:
            float: Mean squared error for the bucket
        """
        if start >= end:
            return 0.0
            
        n = end - start
        sum_x = cum_sum[end] - cum_sum[start]
        sum_x2 = cum_sum2[end] - cum_sum2[start]
        
        mse = sum_x2 - (sum_x ** 2) / n
        
        return mse
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _optimize_mse_dp(cum_sum: np.ndarray, cum_sum2: np.ndarray, 
                        n: int, num_buckets: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dynamic programming optimization for MSE minimization (Numba-accelerated).
        
        This function implements the core DP algorithm with optimized search bounds
        to reduce computational complexity.
        """
        # DP table: dp[i][j] = min MSE for first i points using j buckets
        dp = np.full((n + 1, num_buckets + 1), np.inf)
        dp[0, 0] = 0
        
        # Parent pointers for reconstruction
        parent = np.zeros((n + 1, num_buckets + 1), dtype=np.int32)
        
        # Fill DP table with optimized search bounds
        for i in range(1, n + 1):
            for j in range(1, min(i, num_buckets) + 1):
                # Limit search range for efficiency
                min_k = max(j - 1, (i * (j - 1)) // num_buckets - 10)
                max_k = min(i - 1, (i * j) // num_buckets + 10)
                
                for k in range(min_k, max_k + 1):
                    if k < i:
                        # Calculate MSE for bucket [k, i)
                        bucket_n = i - k
                        sum_x = cum_sum[i] - cum_sum[k]
                        sum_x2 = cum_sum2[i] - cum_sum2[k]
                        mse = sum_x2 - (sum_x ** 2) / bucket_n
                        
                        cost = dp[k, j-1] + mse
                        if cost < dp[i, j]:
                            dp[i, j] = cost
                            parent[i, j] = k
        
        return dp, parent
    
    def _optimize_mse_fast(self, sorted_scores: np.ndarray, 
                          num_buckets: int, *, verbose: bool = True) -> List[float]:
        """
        Find boundaries that minimize MSE using optimized dynamic programming.
        
        This implementation uses precomputed statistics and optional Numba acceleration
        for significant performance improvements on large datasets.
        
        Parameters:
            sorted_scores (np.ndarray): Sorted array of FICO scores
            num_buckets (int): Target number of rating categories
            
        Returns:
            List[float]: Optimal bucket boundaries
        """
        n = len(sorted_scores)
        
        if verbose:
            print(f"\nProcessing {n:,} data points")
            print(f"Target rating categories: {num_buckets}")
        
        # Precompute cumulative statistics
        cum_sum, cum_sum2 = self._precompute_mse_stats(sorted_scores)
        
        if verbose:
            print(f"\nOptimizing bucket boundaries...")
        # Execute dynamic programming
        if HAS_NUMBA:
            dp, parent = self._optimize_mse_dp(cum_sum, cum_sum2, n, num_buckets)
        else:
            # Non-numba version with progress bar
            dp = np.full((n + 1, num_buckets + 1), np.inf)
            dp[0, 0] = 0
            parent = np.zeros((n + 1, num_buckets + 1), dtype=int)
            
            # Create progress bar
            total_iterations = n * num_buckets
            with tqdm(total=total_iterations, desc="Computing", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                     disable=not verbose) as pbar:
                
                for i in range(1, n + 1):
                    for j in range(1, min(i, num_buckets) + 1):
                        # Limit search range
                        min_k = max(j - 1, (i * (j - 1)) // num_buckets - 10)
                        max_k = min(i - 1, (i * j) // num_buckets + 10)
                        
                        for k in range(min_k, max_k + 1):
                            if k < i:
                                mse = self._compute_bucket_mse(cum_sum, cum_sum2, k, i)
                                cost = dp[k, j-1] + mse
                                
                                if cost < dp[i, j]:
                                    dp[i, j] = cost
                                    parent[i, j] = k
                        
                        # Update progress bar
                        pbar.update(1)
                        
                        # Display current optimal value periodically
                        if (i * num_buckets + j) % 1000 == 0:
                            if dp[i, j] != np.inf:
                                pbar.set_postfix({"MSE": f"{dp[i, j]:.2f}"})
        
        # Reconstruct boundaries
        boundaries = []
        i = n
        j = num_buckets
        
        while j > 0:
            k = int(parent[i, j])  # Ensure integer type
            if k > 0 and k < n:
                boundaries.append(sorted_scores[k])
            i = k
            j -= 1
            
        boundaries.reverse()
        
        # Add min and max boundaries
        boundaries = [sorted_scores[0]] + boundaries + [sorted_scores[-1] + 1]
        
        if verbose:
            print(f"\nOptimization complete")
            print(f"Final MSE: {dp[n, num_buckets]:.4f}")
            print(f"Rating boundaries: {', '.join([f'{int(b)}' for b in boundaries[1:-1]])}")
        
        return boundaries
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _optimize_ll_dp(cum_defaults: np.ndarray, n: int, 
                       num_buckets: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dynamic programming optimization for log-likelihood maximization (Numba-accelerated).
        
        This function maximizes the binomial log-likelihood function to find optimal
        risk-based bucket boundaries.
        """
        # DP table: dp[i][j] = max log-likelihood for first i points using j buckets
        dp = np.full((n + 1, num_buckets + 1), -np.inf)
        dp[0, 0] = 0
        
        # Parent pointers
        parent = np.zeros((n + 1, num_buckets + 1), dtype=np.int32)
        
        # Fill DP table with optimized search bounds
        for i in range(1, n + 1):
            for j in range(1, min(i, num_buckets) + 1):
                # Limit search range for efficiency
                min_k = max(j - 1, (i * (j - 1)) // num_buckets - 10)
                max_k = min(i - 1, (i * j) // num_buckets + 10)
                
                for k in range(min_k, max_k + 1):
                    if k < i:
                        # Calculate log-likelihood for bucket [k, i)
                        n_bucket = i - k
                        k_bucket = cum_defaults[i] - cum_defaults[k]
                        
                        # Avoid log(0) with Laplace smoothing
                        if k_bucket == 0:
                            p_bucket = 1e-10
                        elif k_bucket == n_bucket:
                            p_bucket = 1 - 1e-10
                        else:
                            p_bucket = k_bucket / n_bucket
                            
                        ll = (k_bucket * np.log(p_bucket) + 
                              (n_bucket - k_bucket) * np.log(1 - p_bucket))
                        
                        cost = dp[k, j-1] + ll
                        if cost > dp[i, j]:
                            dp[i, j] = cost
                            parent[i, j] = k
        
        return dp, parent
    
    def _optimize_log_likelihood_fast(self, sorted_scores: np.ndarray, 
                                     sorted_defaults: np.ndarray,
                                     num_buckets: int, *, verbose: bool = True) -> List[float]:
        """
        Find boundaries that maximize log-likelihood using optimized dynamic programming.
        
        This method identifies bucket boundaries that maximize the likelihood of observed
        default patterns, resulting in risk-homogeneous rating categories.
        
        Parameters:
            sorted_scores (np.ndarray): Sorted array of FICO scores
            sorted_defaults (np.ndarray): Sorted binary default indicators
            num_buckets (int): Target number of rating categories
            
        Returns:
            List[float]: Optimal bucket boundaries for risk-based segmentation
        """
        n = len(sorted_scores)
        
        if verbose:
            print(f"\nProcessing {n:,} data points")
            print(f"Target rating categories: {num_buckets}")
            print(f"Overall default rate: {np.mean(sorted_defaults):.2%}")
        
        # Precompute cumulative defaults
        cum_defaults = np.concatenate([[0], np.cumsum(sorted_defaults)])
        
        if verbose:
            print(f"\nOptimizing bucket boundaries...")
        # Execute dynamic programming
        if HAS_NUMBA:
            dp, parent = self._optimize_ll_dp(cum_defaults, n, num_buckets)
        else:
            # Non-numba version with progress bar
            dp = np.full((n + 1, num_buckets + 1), -np.inf)
            dp[0, 0] = 0
            parent = np.zeros((n + 1, num_buckets + 1), dtype=int)
            
            # Create progress bar
            total_iterations = n * num_buckets
            with tqdm(total=total_iterations, desc="Computing",
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                     disable=not verbose) as pbar:
                
                for i in range(1, n + 1):
                    for j in range(1, min(i, num_buckets) + 1):
                        # Limit search range for efficiency
                        min_k = max(j - 1, (i * (j - 1)) // num_buckets - 10)
                        max_k = min(i - 1, (i * j) // num_buckets + 10)
                        
                        for k in range(min_k, max_k + 1):
                            if k < i:
                                # Calculate log-likelihood for bucket [k, i)
                                n_bucket = i - k
                                k_bucket = cum_defaults[i] - cum_defaults[k]
                                
                                # Avoid log(0) with Laplace smoothing
                                if k_bucket == 0:
                                    p_bucket = 1e-10
                                elif k_bucket == n_bucket:
                                    p_bucket = 1 - 1e-10
                                else:
                                    p_bucket = k_bucket / n_bucket
                                    
                                ll = (k_bucket * np.log(p_bucket) + 
                                      (n_bucket - k_bucket) * np.log(1 - p_bucket))
                                
                                cost = dp[k, j-1] + ll
                                if cost > dp[i, j]:
                                    dp[i, j] = cost
                                    parent[i, j] = k
                        
                        # Update progress bar
                        pbar.update(1)
                        
                        # Display current optimal value periodically
                        if (i * num_buckets + j) % 1000 == 0:
                            if dp[i, j] != -np.inf:
                                pbar.set_postfix({"LL": f"{dp[i, j]:.2f}"})
        
        # Reconstruct boundaries
        boundaries = []
        i = n
        j = num_buckets
        
        while j > 0:
            k = int(parent[i, j])
            if k > 0 and k < n:
                boundaries.append(sorted_scores[k])
            i = k
            j -= 1
            
        boundaries.reverse()
        
        # Add min and max boundaries
        boundaries = [sorted_scores[0]] + boundaries + [sorted_scores[-1] + 1]
        
        if verbose:
            print(f"\nOptimization complete")
            print(f"Log-likelihood: {dp[n, num_buckets]:.4f}")
            print(f"Rating boundaries: {', '.join([f'{int(b)}' for b in boundaries[1:-1]])}")
        
        return boundaries
    
    def _create_rating_function(self) -> Callable[[float], int]:
        """
        Construct an optimized rating assignment function using binary search.
        
        This method creates a closure that efficiently maps FICO scores to their
        corresponding rating categories using numpy's searchsorted algorithm.
        
        Returns:
            Callable[[float], int]: Vectorized rating assignment function
        """
        # Pre-convert boundaries to numpy array for efficient search
        boundaries_array = np.array(self.boundaries)
        
        def rate_score(fico_score: float) -> int:
            """
            Map a FICO score to its corresponding credit rating.
            
            Uses binary search for O(log n) complexity.
            """
            # Binary search for the appropriate bucket
            idx = np.searchsorted(boundaries_array, fico_score, side='right')
            if idx == 0:
                return self.num_buckets
            elif idx >= len(boundaries_array):
                return 1
            else:
                return self.num_buckets - idx + 1
            
        return rate_score
    
    def transform(self, fico_scores: np.ndarray) -> np.ndarray:
        """
        Transform FICO scores to discrete credit rating categories.
        
        This method applies the fitted quantization model to map continuous credit
        scores into their corresponding rating buckets. The transformation is
        vectorized for efficient batch processing.
        
        Parameters:
            fico_scores (np.ndarray): Array of FICO credit scores to transform
            
        Returns:
            np.ndarray: Array of integer ratings (1=best creditworthiness, 
                       num_buckets=highest risk)
                       
        Raises:
            ValueError: If the quantizer has not been fitted
        """
        if self.rating_map is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Vectorized transformation using searchsorted
        boundaries_array = np.array(self.boundaries)
        indices = np.searchsorted(boundaries_array, fico_scores, side='right')
        
        # Convert indices to ratings
        ratings = self.num_buckets - indices + 1
        ratings = np.clip(ratings, 1, self.num_buckets)
        
        return ratings
    
    def get_bucket_stats(self, fico_scores: np.ndarray, 
                        defaults: np.ndarray = None) -> pd.DataFrame:
        """
        Generate comprehensive statistical summary for each rating category.
        
        This method computes key risk metrics and distributional statistics for
        each rating bucket, providing insights into rating homogeneity and
        risk differentiation.
        
        Parameters:
            fico_scores (np.ndarray): Array of FICO scores for analysis
            defaults (np.ndarray, optional): Binary default indicators for 
                calculating empirical default rates
            
        Returns:
            pd.DataFrame: Statistical summary with columns:
                - Rating: Credit rating category (1 to num_buckets)
                - Count: Number of observations in bucket
                - Min_FICO: Minimum FICO score in bucket
                - Max_FICO: Maximum FICO score in bucket
                - Mean_FICO: Average FICO score in bucket
                - Std_FICO: Standard deviation of FICO scores
                - Default_Rate: Empirical default rate (if defaults provided)
        """
        ratings = self.transform(fico_scores)
        
        stats = []
        for rating in range(1, self.num_buckets + 1):
            mask = ratings == rating
            bucket_scores = fico_scores[mask]
            
            stat_dict = {
                'Rating': rating,
                'Count': len(bucket_scores),
                'Min_FICO': np.min(bucket_scores) if len(bucket_scores) > 0 else np.nan,
                'Max_FICO': np.max(bucket_scores) if len(bucket_scores) > 0 else np.nan,
                'Mean_FICO': np.mean(bucket_scores) if len(bucket_scores) > 0 else np.nan,
                'Std_FICO': np.std(bucket_scores) if len(bucket_scores) > 0 else np.nan
            }
            
            if defaults is not None:
                bucket_defaults = defaults[mask]
                stat_dict['Default_Rate'] = (np.mean(bucket_defaults) 
                                            if len(bucket_defaults) > 0 else np.nan)
                
            stats.append(stat_dict)
            
        return pd.DataFrame(stats)
    
    def plot_quantization(self, fico_scores: np.ndarray, 
                         defaults: np.ndarray = None,
                         figsize: Tuple[int, int] = (12, 8)):
        """
        Generate comprehensive visualization of credit rating quantization results.
        
        This method creates publication-quality plots illustrating the distribution
        of FICO scores across rating categories and the associated risk profiles.
        
        Parameters:
            fico_scores (np.ndarray): Array of FICO scores for visualization
            defaults (np.ndarray, optional): Binary default indicators for risk analysis
            figsize (Tuple[int, int]): Figure dimensions (width, height) in inches
            
        Returns:
            matplotlib.figure.Figure: Figure object containing:
                - Upper panel: FICO score histogram with optimal rating boundaries
                - Lower panel: Default rate by rating category (if defaults provided)
        """
        ratings = self.transform(fico_scores)
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: FICO distribution with boundaries
        ax1 = axes[0]
        ax1.hist(fico_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add vertical lines for boundaries
        for i, boundary in enumerate(self.boundaries[1:-1]):
            ax1.axvline(boundary, color='red', linestyle='--', alpha=0.8)
            
        ax1.set_xlabel('FICO Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('FICO Score Distribution with Rating Boundaries')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Default rates by rating
        if defaults is not None:
            ax2 = axes[1]
            
            default_rates = []
            rating_labels = []
            
            for rating in range(1, self.num_buckets + 1):
                mask = ratings == rating
                if np.sum(mask) > 0:
                    default_rate = np.mean(defaults[mask])
                    default_rates.append(default_rate * 100)
                    rating_labels.append(f'Rating {rating}')
                    
            bars = ax2.bar(rating_labels, default_rates, color='coral', edgecolor='black')
            
            # Add value labels on bars
            for bar, rate in zip(bars, default_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rate:.1f}%', ha='center', va='bottom')
                
            ax2.set_xlabel('Credit Rating')
            ax2.set_ylabel('Default Rate (%)')
            ax2.set_title('Default Rates by Credit Rating')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels if needed
            if len(rating_labels) > 7:
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig


def calculate_expected_loss(fico_score: float, loan_amount: float,
                           quantizer: CreditRatingQuantizer,
                           pd_by_rating: Dict[int, float],
                           recovery_rate: float = 0.1) -> float:
    """
    Calculate expected credit loss (ECL) for a loan using quantized ratings.
    
    This function implements the standard expected loss formula:
    ECL = PD × LGD × EAD
    where:
    - PD (Probability of Default) is determined by the credit rating
    - LGD (Loss Given Default) = 1 - Recovery Rate
    - EAD (Exposure at Default) = Loan Amount
    
    Parameters:
        fico_score (float): Borrower's FICO credit score
        loan_amount (float): Exposure at default (loan principal)
        quantizer (CreditRatingQuantizer): Fitted quantizer for rating assignment
        pd_by_rating (Dict[int, float]): Rating-specific probability of default
        recovery_rate (float): Expected recovery rate in event of default
            (default: 0.1 or 10%, implying 90% LGD)
        
    Returns:
        float: Expected loss amount in currency units
        
    Example:
        >>> quantizer = CreditRatingQuantizer(method='log_likelihood')
        >>> quantizer.fit(fico_scores, defaults)
        >>> pd_map = {1: 0.001, 2: 0.005, ..., 10: 0.20}
        >>> ecl = calculate_expected_loss(720, 100000, quantizer, pd_map, 0.1)
        >>> print(f"Expected Loss: ${ecl:,.2f}")
    """
    rating = quantizer.transform(np.array([fico_score]))[0]
    pd = pd_by_rating.get(rating, 0.5)  # Default to 50% if rating not found
    
    expected_loss = pd * (1 - recovery_rate) * loan_amount
    
    return expected_loss


@njit(cache=True)
def _reconstruct_boundary_indices(parent: np.ndarray, n: int, num_buckets: int) -> np.ndarray:
    """
    Reconstruct internal boundary indices [b1, b2, ..., b_{nb-1}] from parent table.
    Returns an array of shape (num_buckets - 1,) with indices into the sorted array.
    """
    m = num_buckets - 1
    out_idx = np.empty(m, dtype=np.int32)
    i = n
    j = num_buckets
    pos = m - 1
    while j > 0:
        k = int(parent[i, j])
        if j != 1 and k > 0 and k < n:
            out_idx[pos] = k
            pos -= 1
        i = k
        j -= 1
    return out_idx


def _reconstruct_boundaries_from_parent(sorted_scores: np.ndarray, parent: np.ndarray, n: int, num_buckets: int) -> List[float]:
    """Python helper to reconstruct value boundaries from parent indices."""
    idx = _reconstruct_boundary_indices(parent, n, num_buckets)
    boundaries = [sorted_scores[0]]
    for k in idx:
        boundaries.append(sorted_scores[int(k)])
    boundaries.append(sorted_scores[-1] + 1)
    return boundaries


@njit(cache=True, fastmath=True)
def _mse_from_indices(cum_sum: np.ndarray, cum_sum2: np.ndarray, n_total: int, idx: np.ndarray) -> float:
    """
    Compute average MSE given internal boundary indices.
    idx: array of shape (num_buckets-1,) with sorted array split points.
    """
    total_sse = 0.0
    last = 0
    for t in range(idx.shape[0]):
        e = int(idx[t])
        if e > last:
            n = e - last
            sum_x = cum_sum[e] - cum_sum[last]
            sum_x2 = cum_sum2[e] - cum_sum2[last]
            total_sse += sum_x2 - (sum_x * sum_x) / n
        last = e
    # last segment
    e = n_total
    if e > last:
        n = e - last
        sum_x = cum_sum[e] - cum_sum[last]
        sum_x2 = cum_sum2[e] - cum_sum2[last]
        total_sse += sum_x2 - (sum_x * sum_x) / n
    return total_sse / n_total


def evaluate_bucket_counts_fast(
    fico_scores: np.ndarray,
    defaults: np.ndarray,
    bucket_counts: List[int],
) -> Dict[str, List[float]]:
    """
    Efficiently evaluate quantization error across different bucket counts:
    - Single sort pass for inputs
    - Precompute cumulative statistics
    - Directly call accelerated DP cores and compute MSE consistently

    Returns:
      {"mse": [...], "log_likelihood_mse": [...]}  MSE of quantization under two optimization methods
    """
    # Preprocess
    sorted_idx = np.argsort(fico_scores)
    s_scores = fico_scores[sorted_idx]
    s_defaults = defaults[sorted_idx]

    # Precompute
    q = CreditRatingQuantizer(method="mse")
    cum_sum, cum_sum2 = q._precompute_mse_stats(s_scores)
    cum_defaults = np.concatenate([[0], np.cumsum(s_defaults)])

    n = len(s_scores)
    mse_list: List[float] = []
    ll_mse_list: List[float] = []

    for nb in bucket_counts:
        # MSE optimization
        if HAS_NUMBA:
            dp, parent = CreditRatingQuantizer._optimize_mse_dp(cum_sum, cum_sum2, n, nb)
        else:
            # Non-numba fallback (quiet)
            dp = np.full((n + 1, nb + 1), np.inf)
            dp[0, 0] = 0
            parent = np.zeros((n + 1, nb + 1), dtype=int)
            for i in range(1, n + 1):
                for j in range(1, min(i, nb) + 1):
                    min_k = max(j - 1, (i * (j - 1)) // nb - 10)
                    max_k = min(i - 1, (i * j) // nb + 10)
                    for k in range(min_k, max_k + 1):
                        if k < i:
                            bucket_n = i - k
                            sum_x = cum_sum[i] - cum_sum[k]
                            sum_x2 = cum_sum2[i] - cum_sum2[k]
                            mse = sum_x2 - (sum_x ** 2) / bucket_n
                            cost = dp[k, j - 1] + mse
                            if cost < dp[i, j]:
                                dp[i, j] = cost
                                parent[i, j] = k
        if HAS_NUMBA:
            b_idx = _reconstruct_boundary_indices(parent, n, nb)
            mse_val = _mse_from_indices(cum_sum, cum_sum2, n, b_idx)
        else:
            boundaries_mse = _reconstruct_boundaries_from_parent(s_scores, parent, n, nb)
            b_idx = np.searchsorted(s_scores, np.array(boundaries_mse[1:-1]), side="left")
            mse_val = _mse_from_indices(cum_sum, cum_sum2, n, b_idx.astype(np.int32))
        mse_list.append(mse_val)

        # Log-likelihood optimization (measure quantization MSE for comparability)
        if HAS_NUMBA:
            dp_ll, parent_ll = CreditRatingQuantizer._optimize_ll_dp(cum_defaults, n, nb)
        else:
            dp_ll = np.full((n + 1, nb + 1), -np.inf)
            dp_ll[0, 0] = 0
            parent_ll = np.zeros((n + 1, nb + 1), dtype=int)
            for i in range(1, n + 1):
                for j in range(1, min(i, nb) + 1):
                    min_k = max(j - 1, (i * (j - 1)) // nb - 10)
                    max_k = min(i - 1, (i * j) // nb + 10)
                    for k in range(min_k, max_k + 1):
                        if k < i:
                            n_bucket = i - k
                            k_bucket = cum_defaults[i] - cum_defaults[k]
                            if k_bucket == 0:
                                p_bucket = 1e-10
                            elif k_bucket == n_bucket:
                                p_bucket = 1 - 1e-10
                            else:
                                p_bucket = k_bucket / n_bucket
                            ll = k_bucket * np.log(p_bucket) + (n_bucket - k_bucket) * np.log(1 - p_bucket)
                            cost = dp_ll[k, j - 1] + ll
                            if cost > dp_ll[i, j]:
                                dp_ll[i, j] = cost
                                parent_ll[i, j] = k
        if HAS_NUMBA:
            b_idx_ll = _reconstruct_boundary_indices(parent_ll, n, nb)
            ll_mse_val = _mse_from_indices(cum_sum, cum_sum2, n, b_idx_ll)
        else:
            boundaries_ll = _reconstruct_boundaries_from_parent(s_scores, parent_ll, n, nb)
            b_idx_ll = np.searchsorted(s_scores, np.array(boundaries_ll[1:-1]), side="left")
            ll_mse_val = _mse_from_indices(cum_sum, cum_sum2, n, b_idx_ll.astype(np.int32))
        ll_mse_list.append(ll_mse_val)

    return {"mse": mse_list, "log_likelihood_mse": ll_mse_list}


# Performance testing and demonstration
if __name__ == "__main__":
    import time
    
    # Generate synthetic data for testing
    n_samples = 10000
    
    # Create realistic FICO score distribution
    fico_scores = np.concatenate([
        np.random.normal(720, 50, int(n_samples * 0.6)),  # Good credit
        np.random.normal(650, 40, int(n_samples * 0.3)),  # Fair credit
        np.random.normal(580, 30, int(n_samples * 0.1))   # Poor credit
    ])
    
    # Clip to valid FICO range
    fico_scores = np.clip(fico_scores, 300, 850)
    
    # Generate defaults (higher default rate for lower scores)
    default_probs = 1 / (1 + np.exp((fico_scores - 650) / 50))
    defaults = np.random.binomial(1, default_probs)
    
    # Test MSE method with timing
    print("Testing MSE Quantization Method")
    start_time = time.time()
    quantizer_mse = CreditRatingQuantizer(method='mse')
    quantizer_mse.fit(fico_scores, defaults, num_buckets=10)
    mse_time = time.time() - start_time
    print(f"\nTime elapsed: {mse_time:.2f} seconds")
    
    # Test log-likelihood method with timing
    print("\nTesting Log-Likelihood Quantization Method")
    start_time = time.time()
    quantizer_ll = CreditRatingQuantizer(method='log_likelihood')
    quantizer_ll.fit(fico_scores, defaults, num_buckets=10)
    ll_time = time.time() - start_time
    print(f"\nTime elapsed: {ll_time:.2f} seconds")
    
    print(f"\nTotal execution time: {mse_time + ll_time:.2f} seconds")
    
    # Display results
    print("\n\nLog-Likelihood Method - Bucket Statistics:")
    print(quantizer_ll.get_bucket_stats(fico_scores, defaults))
