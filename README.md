# JP Morgan Quantitative Research Tasks

[中文](README_ZH.md) | English

This repository contains solutions to four quantitative research tasks designed for risk management and trading operations. Each task focuses on different aspects of financial modeling, from commodity pricing to credit risk assessment.

## Project Structure

The project is organized into English and Chinese versions, with each task containing requirements, implementations, and example solutions.

```
JP/
├── EN/                             # English version
│   ├── Task 1 - gas_price_analysis/
│   ├── Task 2 - gas_storage_pricing/
│   ├── Task 3 - loan_default_prediction/
│   └── Task 4 - credit_rating_quantization/
└── ZH/                             # Chinese version
    ├── Task 1 - gas_price_analysis/
    ├── Task 2 - gas_storage_pricing 2/
    ├── Task 3 - loan_default_prediction/
    └── Task 4 - credit_rating_quantization/
```

## Task Overview

### Task 1: Natural Gas Price Analysis and Forecasting

**Objective**: Develop a comprehensive natural gas price analysis system that can estimate prices for any given date and extrapolate prices for future periods to support long-term storage contract pricing.

**Key Features**:
- Monthly price data processing from October 2020 to September 2024
- Price estimation function supporting both interpolation and extrapolation
- Pattern recognition for seasonal trends and market factors
- Comprehensive visualizations showing price patterns over time

**Technical Implementation**:
- Time series analysis using historical natural gas price data
- Statistical modeling for price trend identification
- Seasonal decomposition and pattern analysis
- Forecasting algorithms for price extrapolation

### Task 2: Gas Storage Contract Pricing Model

**Objective**: Create a prototype pricing model for gas storage contracts that can handle multiple injection and withdrawal dates with specified volumes.

**Key Features**:
- Monte Carlo simulation with GARCH(1,1) volatility modeling
- Flexible contract structure supporting arbitrary injection/withdrawal schedules
- Physical constraint validation (storage capacity, flow rates)
- Comprehensive cash flow analysis including storage costs

**Technical Implementation**:
- Price simulation using advanced volatility modeling
- Constraint optimization for storage operations
- Net present value calculation for contract valuation
- Scenario analysis for risk assessment

### Task 3: Loan Default Prediction Model

**Objective**: Build a predictive model that estimates the probability of loan default and calculates expected loss based on borrower characteristics.

**Key Features**:
- Machine learning models for default probability estimation
- Expected loss calculation with 10% recovery rate assumption
- Multiple modeling approaches with comparative analysis
- Feature engineering using borrower financial metrics

**Data Variables**:
- Customer demographics and employment history
- Credit lines and debt outstanding
- FICO scores and income levels
- Historical default indicators

### Task 4: Credit Rating Quantization

**Objective**: Develop a quantization system that maps FICO scores to credit ratings using optimal bucket boundaries that minimize error or maximize likelihood.

**Key Features**:
- Dynamic programming approach for optimal bucket boundaries
- Two optimization methods: Mean Squared Error and Log-Likelihood maximization
- Monotonic rating system where lower ratings indicate better credit scores
- Generalizable approach for future datasets

**Technical Approaches**:
- MSE minimization for approximation-based quantization
- Log-likelihood maximization considering default density
- Dynamic programming for computational efficiency
- Statistical validation of bucket distributions

## Technical Stack

- **Python**: Primary programming language for all implementations
- **Jupyter Notebooks**: Interactive analysis and visualization
- **NumPy/Pandas**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning models and evaluation
- **Matplotlib/Seaborn**: Data visualization and plotting
- **GARCH Models**: Volatility estimation for price simulation
- **Dynamic Programming**: Optimization algorithms for quantization

## Data Sources

- Natural gas price data (CSV format, monthly snapshots)
- Loan performance data with borrower characteristics
- FICO scores and default indicators
- Market data for volatility estimation

## Getting Started

1. Navigate to any task directory (EN/ or ZH/)
2. Review the requirements document
3. Execute the Jupyter notebook or Python scripts
4. Compare results with provided example answers

## Key Assumptions

- Zero interest rates for present value calculations
- No transport delays for gas storage operations
- Market holidays and weekends are not considered
- Recovery rate of 10% for loan default calculations

## Model Validation

All models include comprehensive testing scenarios to ensure accuracy and robustness. Each implementation provides validation against sample inputs and comparison with expected outcomes.

## Future Enhancements

- Integration of real-time market data feeds
- Advanced volatility modeling with regime-switching
- Interest rate term structure incorporation
- Extended feature engineering for credit models
- Performance optimization for large-scale datasets

## Contributing

This project represents completed solutions to quantitative research challenges. Each task demonstrates different aspects of financial modeling and risk management in trading operations.

Pull requests are welcome! Feel free to contribute improvements, bug fixes, or additional features.

## Author

**Curtis Yan**  
Email: realthat@foxmail.com  
Year: 2025
