# Gas Storage Contract Pricing Model Requirements

## Overview
You need to create a prototype pricing model for gas storage contracts that can undergo further validation and testing before production deployment. While this model may eventually serve as the foundation for fully automated client quoting, it will initially be used by the trading desk with manual oversight to explore options with clients.

## Objective
Develop a function that utilizes previously created data to price gas storage contracts. The solution should accommodate clients who may want to select multiple dates for gas injection and withdrawal with specified volumes. Your approach should generalize the previous explanation and consider all cash flows involved in the product.

## Input Parameters
The pricing function should accept the following inputs:

1. **Injection dates** - Dates when gas will be injected into storage
2. **Withdrawal dates** - Dates when gas will be withdrawn from storage
3. **Commodity prices** - Purchase/sale prices on the specified injection/withdrawal dates
4. **Flow rates** - The rate at which gas can be injected/withdrawn
5. **Maximum storage capacity** - The maximum volume that can be stored
6. **Storage costs** - Costs associated with storing the gas

## Assumptions
- No transport delay
- Zero interest rates
- Market holidays, weekends, and bank holidays are not considered

## Deliverables
1. A function that takes the above inputs and returns the contract value
2. Test cases with sample inputs to validate the pricing model

## Testing
Test your implementation by selecting various sample inputs to ensure the model handles different scenarios correctly.
