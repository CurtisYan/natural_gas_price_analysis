# Natural Gas Price Data Analysis and Forecasting Requirements

## Background
After investigating the source of existing data, you have discovered that the current process involves taking monthly snapshots of prices from a market data provider. These snapshots represent the market price of natural gas delivered at the end of each calendar month. The data is available for approximately the next 18 months and is combined with historical prices in a time series database. You have been granted access and can download the data in CSV format.

## Objective
Use the monthly snapshot data to create a comprehensive view of existing price patterns and develop extrapolations for an additional year to support indicative pricing for longer-term storage contracts that clients may require.

## Data Specifications
- **Data Source**: Monthly natural gas price data in CSV format
- **Time Period**: October 31, 2020 to September 30, 2024
- **Data Points**: Each point represents the purchase price of natural gas at the end of a month

## Requirements

### 1. Data Analysis
- Download and process the monthly natural gas price data
- Analyze the data to identify patterns and trends
- Develop a model to estimate the purchase price of gas at any given date

### 2. Price Estimation Function
- Create a function that takes a date as input
- Return a price estimate for the given date
- Support both historical dates (interpolation) and future dates (extrapolation)

### 3. Forecasting
- Extrapolate prices for one year beyond the available data
- Ensure the extrapolation method captures identified patterns

### 4. Visualization
- Create visualizations to identify patterns in the data
- Explore seasonal trends by analyzing month-of-year effects
- Consider and document factors that might influence natural gas price variations

## Assumptions
- Market holidays, weekends, and bank holidays need not be accounted for
- Focus on monthly and seasonal patterns rather than daily fluctuations

## Deliverables
1. Code to download and process the CSV data
2. Price estimation function with date input/price output capability
3. Visualizations showing price patterns and trends
4. Documentation of identified factors affecting gas prices
5. Complete, tested code implementation
