# Credit Rating Quantization Requirements

## Overview
Charlie wants to make her model work for future data sets, so she needs a general approach to generating the buckets. Given a set number of buckets corresponding to the number of input labels for the model, she would like to find out the boundaries that best summarize the data. You need to create a rating map that maps the FICO score of the borrowers to a rating where a lower rating signifies a better credit score.

The process of doing this is known as quantization. You could consider many ways of solving the problem by optimizing different properties of the resulting buckets, such as the mean squared error or log-likelihood (see below for definitions). For background on quantization, see [here](https://en.wikipedia.org/wiki/Quantization_(signal_processing)).

## Objective
Develop a quantization method that maps FICO scores to credit ratings, where lower ratings represent better credit scores.

## Technical Approaches

### Mean Squared Error (MSE)
You can view this question as an approximation problem and try to map all the entries in a bucket to one value, minimizing the associated squared error. We are now looking to minimize the following:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2
$$
Where:
- $Y_i$ is the original FICO score
- $\hat{Y}_i$ is the assigned bucket value
- $n$ is the number of data points

### Log-likelihood
A more sophisticated possibility is to maximize the following log-likelihood function:

$$
LL(b_1, ..., b_{r-1}) = \sum_{i=1}^{r} [k_i \ln p_i + (n_i - k_i) \ln(1 - p_i)]
$$
Where:
- $b_i$ is the bucket boundaries
- $n_i$ is the number of records in each bucket
- $k_i$ is the number of defaults in each bucket
- $p_i = k_i / n_i$ is the probability of default in the bucket

This function considers how rough the discretization is and the density of defaults in each bucket. This problem could be addressed by splitting it into subproblems, which can be solved incrementally (i.e., through a dynamic programming approach). For example, you can break the problem into two subproblems, creating five buckets for FICO scores ranging from 0 to 600 and five buckets for FICO scores ranging from 600 to 850. Refer to this [page](link) for more context behind a likelihood function. This [page](link) may also be helpful for background on dynamic programming.

## Requirements

1. **Input**: 
   - FICO scores from the loan data
   - Number of desired buckets/ratings
   
2. **Output**:
   - Rating mapping function that converts FICO scores to ratings
   - Bucket boundaries for each rating
   - Lower rating values should indicate better credit scores

3. **Constraints**:
   - The approach should be generalizable to future datasets
   - The rating system should be monotonic (higher FICO scores → lower ratings)
   - Bucket boundaries should be optimal according to the chosen optimization criteria

## Deliverables

1. **Quantization Function**: A function that takes:
   - Array of FICO scores
   - Number of buckets
   - Optimization method (MSE or log-likelihood)
   - Returns: bucket boundaries and rating mapping

2. **Rating Mapper**: A function that:
   - Takes a FICO score as input
   - Returns the corresponding rating based on the quantization

3. **Evaluation Metrics**:
   - MSE for the quantization
   - Log-likelihood value (if using that approach)
   - Distribution of data points across buckets

4. **Visualization**:
   - Histogram showing FICO score distribution with bucket boundaries
   - Default rates per bucket/rating

## Example Usage

```python
# Example function signature
def quantize_fico_scores(fico_scores, num_buckets, method='mse'):
    """
    Quantize FICO scores into buckets
    
    Parameters:
    - fico_scores: array of FICO scores
    - num_buckets: number of desired buckets
    - method: 'mse' or 'log_likelihood'
    
    Returns:
    - boundaries: list of bucket boundaries
    - rating_map: function that maps FICO score to rating
    """
    pass

# Usage
boundaries, rating_map = quantize_fico_scores(fico_data, 10, method='mse')
rating = rating_map(750)  # Returns rating for FICO score 750
```

## Technical Considerations

1. **Dynamic Programming**: For the log-likelihood approach, consider using dynamic programming to find optimal bucket boundaries efficiently.

2. **Monotonicity**: Ensure that the rating system maintains proper ordering (higher FICO → lower rating).

3. **Edge Cases**: Handle edge cases such as:
   - FICO scores outside the training data range
   - Empty buckets
   - Extreme distributions

4. **Scalability**: The solution should work efficiently for large datasets.
