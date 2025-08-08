"""
信用评级量化模块
作者: Curtis Yan
日期: 2025.8.7

本模块使用量化技术将FICO评分映射到信用评级。
评级数字越小表示信用越好（如：1=最优，10=最差）。

优化改进：
1. 预计算累积和避免重复计算
2. 优化动态规划实现
3. 减少内存分配和数组操作
"""

import platform
import warnings
from typing import List, Tuple, Callable, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Try to import numba for JIT compilation
try:
    from numba import njit, prange

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

# Set Chinese fonts for matplotlib

if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS"]
elif platform.system() == "Windows":
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class CreditRatingQuantizer:
    """
    Maps FICO scores to discrete credit ratings through optimal quantization.

    Supports two optimization methods:
    1. MSE (Mean Squared Error): Minimizes squared error between original and
       quantized scores
    2. Log-likelihood: Maximizes likelihood function considering default
       probabilities
    """

    def __init__(self, method="mse"):
        """
        Initialize the quantizer with specified optimization method.

        Parameters:
            method: 'mse' or 'log_likelihood'
        """
        self.method = method
        self.boundaries = None
        self.rating_map = None
        self.num_buckets = None

    def fit(
        self,
        fico_scores: np.ndarray,
        defaults: np.ndarray = None,
        num_buckets: int = 10,
    ) -> "CreditRatingQuantizer":
        """
        Find optimal bucket boundaries for given FICO scores.

        Parameters:
            fico_scores: Array of FICO scores (300-850)
            defaults: Binary default indicators array (required for log-likelihood
                method)
            num_buckets: Number of rating categories

        Returns:
            self: Fitted quantizer instance
        """
        self.num_buckets = num_buckets

        # Sort for processing
        sorted_indices = np.argsort(fico_scores)
        sorted_scores = fico_scores[sorted_indices]

        if self.method == "mse":
            self.boundaries = self._optimize_mse_fast(sorted_scores, num_buckets)
        elif self.method == "log_likelihood":
            if defaults is None:
                raise ValueError("Log-likelihood method requires default data")
            sorted_defaults = defaults[sorted_indices]
            self.boundaries = self._optimize_log_likelihood_fast(
                sorted_scores, sorted_defaults, num_buckets
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Create rating mapping function
        self.rating_map = self._create_rating_function()

        return self

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _precompute_mse_stats(
        sorted_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预计算累积和与平方累积和（为 MSE 计算提速）。
        注意：实现为静态函数以兼容 Numba（不带 self）。
        """
        n = len(sorted_scores)
        cum_sum = np.zeros(n + 1)
        cum_sum2 = np.zeros(n + 1)

        cum_sum[1:] = np.cumsum(sorted_scores)
        cum_sum2[1:] = np.cumsum(sorted_scores**2)

        return cum_sum, cum_sum2

    def _compute_bucket_mse(
        self, cum_sum: np.ndarray, cum_sum2: np.ndarray, start: int, end: int
    ) -> float:
        """
        Compute bucket MSE quickly using precomputed cumulative sums.
        """
        if start >= end:
            return 0.0

        n = end - start
        sum_x = cum_sum[end] - cum_sum[start]
        sum_x2 = cum_sum2[end] - cum_sum2[start]

        # MSE = E[X^2] - (E[X])^2
        # = sum(x^2)/n - (sum(x)/n)^2
        mse = sum_x2 - (sum_x**2) / n

        return mse

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _optimize_mse_dp(
        cum_sum: np.ndarray, cum_sum2: np.ndarray, n: int, num_buckets: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find boundaries that minimize MSE using dynamic programming
        (numba-accelerated version).
        """
        # Dynamic programming table
        dp = np.full((n + 1, num_buckets + 1), np.inf)
        dp[0, 0] = 0

        # Parent pointers
        parent = np.zeros((n + 1, num_buckets + 1), dtype=np.int32)

        # Fill DP table
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
                        mse = sum_x2 - (sum_x**2) / bucket_n

                        cost = dp[k, j - 1] + mse
                        if cost < dp[i, j]:
                            dp[i, j] = cost
                            parent[i, j] = k

        return dp, parent

    def _optimize_mse_fast(
        self, sorted_scores: np.ndarray, num_buckets: int, *, verbose: bool = True
    ) -> List[float]:
        """
        Find boundaries that minimize MSE using optimized dynamic programming.
        """
        n = len(sorted_scores)

        if verbose:
            print(f"\n处理数据点: {n}个")
            print(f"目标评级数: {num_buckets}个")

        # Precompute cumulative statistics
        cum_sum, cum_sum2 = self._precompute_mse_stats(sorted_scores)

        if verbose:
            print("\n正在计算最优分组...")
        # Execute dynamic programming
        if HAS_NUMBA:
            dp, parent = self._optimize_mse_dp(cum_sum, cum_sum2, n, num_buckets)
        else:
            # Non-numba version
            dp = np.full((n + 1, num_buckets + 1), np.inf)
            dp[0, 0] = 0
            parent = np.zeros((n + 1, num_buckets + 1), dtype=int)

            # Create progress bar
            total_iterations = n * num_buckets
            with tqdm(
                total=total_iterations,
                desc="计算中",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                disable=not verbose,
            ) as pbar:

                for i in range(1, n + 1):
                    for j in range(1, min(i, num_buckets) + 1):
                        # Limit search range
                        min_k = max(j - 1, (i * (j - 1)) // num_buckets - 10)
                        max_k = min(i - 1, (i * j) // num_buckets + 10)

                        for k in range(min_k, max_k + 1):
                            if k < i:
                                mse = self._compute_bucket_mse(cum_sum, cum_sum2, k, i)
                                cost = dp[k, j - 1] + mse

                                if cost < dp[i, j]:
                                    dp[i, j] = cost
                                    parent[i, j] = k

                        # Update progress bar
                        if verbose:
                            pbar.update(1)

                            # Show current optimal value periodically
                            if (i * num_buckets + j) % 1000 == 0:
                                if dp[i, j] != np.inf:
                                    pbar.set_postfix({"MSE": f"{dp[i, j]:.2f}"})

        # Reconstruct boundaries
        boundaries = []
        i = n
        j = num_buckets

        while j > 0:
            k = int(parent[i, j])  # Ensure integer
            if k > 0 and k < n:
                boundaries.append(sorted_scores[k])
            i = k
            j -= 1

        boundaries.reverse()

        # Add min and max boundaries
        boundaries = [sorted_scores[0]] + boundaries + [sorted_scores[-1] + 1]

        if verbose:
            print("\n计算完成")
            print(f"MSE值: {dp[n, num_buckets]:.4f}")
            print(f"评级分界点: {', '.join([f'{int(b)}' for b in boundaries[1:-1]])}")

        return boundaries

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _optimize_ll_dp(
        cum_defaults: np.ndarray, n: int, num_buckets: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find boundaries that maximize log-likelihood using dynamic programming
        (numba-accelerated version).
        """
        # DP table
        dp = np.full((n + 1, num_buckets + 1), -np.inf)
        dp[0, 0] = 0

        # Parent pointers
        parent = np.zeros((n + 1, num_buckets + 1), dtype=np.int32)

        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, min(i, num_buckets) + 1):
                # Limit search range
                min_k = max(j - 1, (i * (j - 1)) // num_buckets - 10)
                max_k = min(i - 1, (i * j) // num_buckets + 10)

                for k in range(min_k, max_k + 1):
                    if k < i:
                        # Calculate log-likelihood for bucket [k, i)
                        n_bucket = i - k
                        k_bucket = cum_defaults[i] - cum_defaults[k]

                        # Avoid log(0)
                        if k_bucket == 0:
                            p_bucket = 1e-10
                        elif k_bucket == n_bucket:
                            p_bucket = 1 - 1e-10
                        else:
                            p_bucket = k_bucket / n_bucket

                        ll = k_bucket * np.log(p_bucket) + (
                            n_bucket - k_bucket
                        ) * np.log(1 - p_bucket)

                        cost = dp[k, j - 1] + ll
                        if cost > dp[i, j]:
                            dp[i, j] = cost
                            parent[i, j] = k

        return dp, parent

    def _optimize_log_likelihood_fast(
        self, sorted_scores: np.ndarray, sorted_defaults: np.ndarray, num_buckets: int, *, verbose: bool = True
    ) -> List[float]:
        """
        Find boundaries that maximize log-likelihood using optimized dynamic
        programming.
        """
        n = len(sorted_scores)

        if verbose:
            print(f"\n处理数据点: {n}个")
            print(f"目标评级数: {num_buckets}个")
            print(f"整体违约率: {np.mean(sorted_defaults):.2%}")

        # Precompute cumulative defaults
        cum_defaults = np.concatenate([[0], np.cumsum(sorted_defaults)])

        if verbose:
            print("\n正在计算最优分组...")
        # Execute dynamic programming
        if HAS_NUMBA:
            dp, parent = self._optimize_ll_dp(cum_defaults, n, num_buckets)
        else:
            # Non-numba version
            dp = np.full((n + 1, num_buckets + 1), -np.inf)
            dp[0, 0] = 0
            parent = np.zeros((n + 1, num_buckets + 1), dtype=int)

            # Create progress bar
            total_iterations = n * num_buckets
            with tqdm(
                total=total_iterations,
                desc="计算中",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                disable=not verbose,
            ) as pbar:

                for i in range(1, n + 1):
                    for j in range(1, min(i, num_buckets) + 1):
                        # Limit search range
                        min_k = max(j - 1, (i * (j - 1)) // num_buckets - 10)
                        max_k = min(i - 1, (i * j) // num_buckets + 10)

                        for k in range(min_k, max_k + 1):
                            if k < i:
                                # Calculate log-likelihood for bucket [k, i)
                                n_bucket = i - k
                                k_bucket = cum_defaults[i] - cum_defaults[k]

                                # Avoid log(0)
                                if k_bucket == 0:
                                    p_bucket = 1e-10
                                elif k_bucket == n_bucket:
                                    p_bucket = 1 - 1e-10
                                else:
                                    p_bucket = k_bucket / n_bucket

                                ll = k_bucket * np.log(p_bucket) + (
                                    n_bucket - k_bucket
                                ) * np.log(1 - p_bucket)

                                cost = dp[k, j - 1] + ll
                                if cost > dp[i, j]:
                                    dp[i, j] = cost
                                    parent[i, j] = k

                        # Update progress bar
                        if verbose:
                            pbar.update(1)

                            # Show current optimal value periodically
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
            print("\n计算完成")
            print(f"对数似然值: {dp[n, num_buckets]:.4f}")
            print(f"评级分界点: {', '.join([f'{int(b)}' for b in boundaries[1:-1]])}")

        return boundaries

    def _create_rating_function(self) -> Callable[[float], int]:
        """
        Create a function that maps FICO scores to ratings.

        Returns a function that assigns ratings based on boundaries.
        """
        # Use binary search to optimize rating lookup
        boundaries_array = np.array(self.boundaries)

        def rate_score(fico_score: float) -> int:
            """Map a single FICO score to its rating."""
            # Use binary search to find rating
            idx = np.searchsorted(boundaries_array, fico_score, side="right")
            if idx == 0:
                return self.num_buckets
            elif idx >= len(boundaries_array):
                return 1
            else:
                return self.num_buckets - idx + 1

        return rate_score

    def transform(self, fico_scores: np.ndarray) -> np.ndarray:
        """
        Transform FICO scores to ratings.

        Parameters:
            fico_scores: Array of FICO scores

        Returns:
            Array of ratings (1=best, num_buckets=worst)
        """
        if self.rating_map is None:
            raise ValueError("Model not fitted. Please call fit() method first.")

        # Vectorized rating conversion
        boundaries_array = np.array(self.boundaries)
        indices = np.searchsorted(boundaries_array, fico_scores, side="right")

        # Convert to ratings
        ratings = self.num_buckets - indices + 1
        ratings = np.clip(ratings, 1, self.num_buckets)

        return ratings

    def get_bucket_stats(
        self, fico_scores: np.ndarray, defaults: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Calculate statistics for each rating bucket.

        Parameters:
            fico_scores: Array of FICO scores
            defaults: Binary default indicators array

        Returns:
            DataFrame containing bucket statistics
        """
        ratings = self.transform(fico_scores)

        stats = []
        for rating in range(1, self.num_buckets + 1):
            mask = ratings == rating
            bucket_scores = fico_scores[mask]

            stat_dict = {
                "评级": rating,
                "样本数": len(bucket_scores),
                "最小FICO": np.min(bucket_scores) if len(bucket_scores) > 0 else np.nan,
                "最大FICO": np.max(bucket_scores) if len(bucket_scores) > 0 else np.nan,
                "平均FICO": (
                    np.mean(bucket_scores) if len(bucket_scores) > 0 else np.nan
                ),
                "标准差": np.std(bucket_scores) if len(bucket_scores) > 0 else np.nan,
            }

            if defaults is not None:
                bucket_defaults = defaults[mask]
                stat_dict["违约率"] = (
                    np.mean(bucket_defaults) if len(bucket_defaults) > 0 else np.nan
                )

            stats.append(stat_dict)

        return pd.DataFrame(stats)

    def plot_quantization(
        self,
        fico_scores: np.ndarray,
        defaults: np.ndarray = None,
        figsize: Tuple[int, int] = (12, 8),
    ):
        """
        Visualize quantization results.

        Creates subplots showing:
        1. FICO score distribution with boundaries
        2. Default rates by bucket (if default data provided)
        """
        ratings = self.transform(fico_scores)

        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # Plot 1: FICO distribution and boundaries
        ax1 = axes[0]
        ax1.hist(fico_scores, bins=50, alpha=0.7, color="skyblue", edgecolor="black")

        # Add boundary vertical lines
        for i, boundary in enumerate(self.boundaries[1:-1]):
            ax1.axvline(boundary, color="red", linestyle="--", alpha=0.8)

        ax1.set_xlabel("FICO评分")
        ax1.set_ylabel("频数")
        ax1.set_title("FICO评分分布及评级边界")
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
                    rating_labels.append(f"评级{rating}")

            bars = ax2.bar(
                rating_labels, default_rates, color="coral", edgecolor="black"
            )

            # Add value labels on bars
            for bar, rate in zip(bars, default_rates):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                )

            ax2.set_xlabel("信用评级")
            ax2.set_ylabel("违约率 (%)")
            ax2.set_title("各信用评级违约率分布")
            ax2.grid(True, alpha=0.3, axis="y")

            # Rotate labels if too many
            if len(rating_labels) > 7:
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        return fig


def calculate_expected_loss(
    fico_score: float,
    loan_amount: float,
    quantizer: CreditRatingQuantizer,
    default_prob_by_rating: Dict[int, float],
    recovery_rate: float = 0.1,
) -> float:
    """
    Calculate expected loss for a loan.

    Parameters:
        fico_score: Borrower's FICO score
        loan_amount: Loan principal amount
        quantizer: Fitted quantizer model
        default_prob_by_rating: Dictionary mapping ratings to default probabilities
        recovery_rate: Expected recovery rate (default 10%)

    Returns:
        Expected loss amount
    """
    rating = quantizer.transform(np.array([fico_score]))[0]
    prob_default = default_prob_by_rating.get(rating, 0.5)  # Default to 50% if rating not found

    expected_loss = prob_default * (1 - recovery_rate) * loan_amount

    return expected_loss


@njit(cache=True)
def _reconstruct_boundary_indices(parent: np.ndarray, n: int, num_buckets: int) -> np.ndarray:
    """
    Reconstruct internal boundary indices [b1, b2, ..., b_{nb-1}] from parent table.
    Returns an array of length (num_buckets - 1) with indices into the sorted array.
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
    # Keep a Python helper for compatibility where list is desired
    idx = _reconstruct_boundary_indices(parent, n, num_buckets)
    boundaries = [sorted_scores[0]]
    for k in idx:
        boundaries.append(sorted_scores[int(k)])
    boundaries.append(sorted_scores[-1] + 1)
    return boundaries


@njit(cache=True, fastmath=True)
def _mse_from_indices(cum_sum: np.ndarray, cum_sum2: np.ndarray, n_total: int, idx: np.ndarray) -> float:
    """
    Compute MSE given internal boundary indices.
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
    高效评估不同桶数下的量化误差：
    - 仅排序一次
    - 预计算累积统计
    - 直接调用加速的DP核心，避免多余的打印/进度条和对象构造

    返回:
      {"mse": [...], "log_likelihood_mse": [...]}  两种方法各自对应的MSE误差
    """
    # 预处理
    sorted_idx = np.argsort(fico_scores)
    s_scores = fico_scores[sorted_idx]
    s_defaults = defaults[sorted_idx]

    # 预计算
    q = CreditRatingQuantizer(method="mse")
    cum_sum, cum_sum2 = q._precompute_mse_stats(s_scores)
    cum_defaults = np.concatenate([[0], np.cumsum(s_defaults)])

    n = len(s_scores)
    mse_list: List[float] = []
    ll_mse_list: List[float] = []

    for nb in bucket_counts:
        # MSE方法
        if HAS_NUMBA:
            dp, parent = CreditRatingQuantizer._optimize_mse_dp(cum_sum, cum_sum2, n, nb)
        else:
            # 退化到非numba实现（无打印）
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
            # Fallback: compute via indices built from values for consistency
            b_idx = np.searchsorted(s_scores, np.array(boundaries_mse[1:-1]), side="left")
            mse_val = _mse_from_indices(cum_sum, cum_sum2, n, b_idx.astype(np.int32))
        mse_list.append(mse_val)

        # 对数似然方法（仍然用MSE来衡量量化误差，便于比较）
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


# Test optimization performance
if __name__ == "__main__":
    import time

    # Generate test data
    np.random.seed(42)
    n_samples = 10000

    fico_scores = np.concatenate(
        [
            np.random.normal(720, 50, int(n_samples * 0.6)),
            np.random.normal(650, 40, int(n_samples * 0.3)),
            np.random.normal(580, 30, int(n_samples * 0.1)),
        ]
    )
    fico_scores = np.clip(fico_scores, 300, 850)

    default_probs = 1 / (1 + np.exp((fico_scores - 650) / 50))
    defaults = np.random.binomial(1, default_probs)

    # Test MSE method performance
    print("测试MSE量化方法")
    start_time = time.time()
    quantizer_mse = CreditRatingQuantizer(method="mse")
    quantizer_mse.fit(fico_scores, defaults, num_buckets=10)
    mse_time = time.time() - start_time
    print(f"\n耗时: {mse_time:.2f}秒")

    # Test log-likelihood method performance
    print("\n测试对数似然量化方法")
    start_time = time.time()
    quantizer_ll = CreditRatingQuantizer(method="log_likelihood")
    quantizer_ll.fit(fico_scores, defaults, num_buckets=10)
    ll_time = time.time() - start_time
    print(f"\n耗时: {ll_time:.2f}秒")

    print(f"\n总执行时间: {mse_time + ll_time:.2f}秒")

    # Display results
    print("\n\n对数似然方法 - 桶统计信息:")
    print(quantizer_ll.get_bucket_stats(fico_scores, defaults))
