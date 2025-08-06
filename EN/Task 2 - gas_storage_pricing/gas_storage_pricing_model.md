# Gas Storage Contract Pricing Model Documentation

## Overview
This pricing model is designed to value natural gas storage contracts by considering all relevant cash flows including injection costs, withdrawal revenues, and storage costs. The model uses Monte Carlo simulation with GARCH(1,1) for price forecasting.

## Core Features

### 1. Price Simulation Function
- **Function**: `simulate_future_prices()`
- **Method**: GARCH(1,1) model for volatility estimation
- **Output**: Multiple price paths for Monte Carlo simulation
- **Based on**: Historical price data analysis

### 2. Contract Valuation Function
- **Function**: `value_gas_storage_contract()`
- **Cash flows considered**:
  - Purchase costs at injection dates
  - Sales revenues at withdrawal dates
  - Daily storage costs
  - Maximum storage capacity constraints
  - Injection/withdrawal rate limits

### 3. Key Constraints
- Maximum storage capacity check
- Injection/withdrawal rate verification
- Storage duration calculation
- Cash flow timing optimization

## Input Parameters

| Parameter | Description | Type |
|-----------|-------------|------|
| `injection_dates` | List of injection dates | List[datetime] |
| `withdrawal_dates` | List of withdrawal dates | List[datetime] |
| `injection_prices` | Prices at injection (fixed or simulated) | Array |
| `withdrawal_prices` | Prices at withdrawal (fixed or simulated) | Array |
| `injection_rate` | Daily injection rate | Float |
| `withdrawal_rate` | Daily withdrawal rate | Float |
| `max_storage` | Maximum storage capacity | Float |
| `storage_cost_per_day` | Daily storage cost per unit | Float |
| `gas_amounts` | Gas amounts for each operation | List[Float] |

## Output
- **Primary**: Total contract value (NPV of all cash flows)
- **Secondary**: Validation messages for constraint violations

## Use Cases

### 1. Seasonal Arbitrage
- Buy in summer (low prices)
- Sell in winter (high prices)
- Optimize storage duration

### 2. Multiple Operations
- Complex injection/withdrawal schedules
- Portfolio optimization
- Risk management

### 3. Quick Turnaround
- Short-term price differentials
- Minimize storage costs
- Maximize turnover

## Model Advantages
- **Flexibility**: Handles arbitrary number of operations
- **Realism**: Incorporates physical constraints
- **Comprehensiveness**: Includes all relevant costs
- **Uncertainty**: Price simulation capability

## Example Usage
```python
# Simple seasonal strategy
contract_value = value_gas_storage_contract(
    injection_dates=[datetime(2024, 6, 1)],
    withdrawal_dates=[datetime(2024, 12, 1)],
    injection_prices=np.array([10.5]),
    withdrawal_prices=np.array([12.8]),
    injection_rate=1000,
    withdrawal_rate=1500,
    max_storage=100000,
    storage_cost_per_day=0.01,
    gas_amounts=[50000]
)
```

## Technical Implementation

### 1. Price Simulation
- Calculate returns from historical data
- Estimate conditional volatility using GARCH
- Generate future price paths

### 2. Constraint Validation
- Check storage capacity limits
- Verify operation timing feasibility
- Ensure accurate cash flow calculations

### 3. Value Calculation
- Calculate all cash flows
- Consider time value (current version assumes zero interest rate)
- Return net present value

## Model Limitations
- Assumes no transport delays
- Interest rate set to zero (extensible)
- Does not consider market holidays
- Price simulation based on historical data

## Future Enhancements
1. Include interest rate term structure
2. Consider transportation costs and delays
3. Add more sophisticated price models
4. Optimize algorithms for better performance
5. Add real-time market data integration

## Contact Information
For questions or improvements, please contact the development team.

## Version
- Version: 1.0
- Last Updated: 2024
