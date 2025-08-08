# Loan Default Prediction Model Requirements

## Overview
The risk manager has collected data on the loan borrowers. The data is in tabular format, with each row providing details of the borrower, including their income, total loans outstanding, and a few other metrics. There is also a column indicating if the borrower has previously defaulted on a loan. You must use this data to build a model that, given details for any loan described above, will predict the probability that the borrower will default (also known as PD: the probability of default). Use the provided data to train a function that will estimate the probability of default for a borrower. Assuming a recovery rate of 10%, this can be used to give the expected loss on a loan.

## Objective
Develop a function that can take in the properties of a loan and output the expected loss.

## Data
The data file `Task 3 and 4_Loan_Data.csv` contains:
- customer_id
- credit_lines_outstanding
- loan_amt_outstanding
- total_debt_outstanding
- income
- years_employed
- fico_score
- default (0 = no default, 1 = default)

## Requirements
1. Build a model to predict the probability of default (PD) for a borrower
2. Create a function that calculates expected loss using:
   - Expected Loss = PD × (1 - Recovery Rate) × Loan Amount
   - Recovery Rate = 10%

## Techniques
You can explore any technique ranging from a simple regression or a decision tree to something more advanced. You can also use multiple methods and provide a comparative analysis.

## Deliverables
1. A function that takes loan properties and outputs the expected loss
2. Model comparison if multiple approaches are used
3. Test cases with sample inputs
