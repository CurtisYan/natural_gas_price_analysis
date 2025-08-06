"""
Gas Storage Contract Pricing Model
Author: Curtis Yan
Date: 2025.8.6

This model calculates the value of a gas storage contract, considering injection, withdrawal, and storage costs.
"""

import numpy as np
from datetime import datetime, timedelta

class GasStorageContractEN:
    def __init__(self, max_storage, injection_rate, withdrawal_rate, storage_cost_per_day):
        self.max_storage = max_storage
        self.injection_rate = injection_rate
        self.withdrawal_rate = withdrawal_rate
        self.storage_cost_per_day = storage_cost_per_day
        self.current_storage = 0
        self.operations = []

    def inject_gas(self, amount, date, price):
        if self.current_storage + amount > self.max_storage:
            raise ValueError("Exceeding maximum storage capacity")
        self.current_storage += amount
        self.operations.append({'date': date, 'amount': amount, 'price': price, 'type': 'inject'})
        return f"Injection successful: {amount} units at {price}"

    def withdraw_gas(self, amount, date, price):
        if self.current_storage < amount:
            raise ValueError("Insufficient storage for withdrawal")
        self.current_storage -= amount
        self.operations.append({'date': date, 'amount': amount, 'price': price, 'type': 'withdraw'})
        return f"Withdrawal successful: {amount} units at {price}"

    def calculate_storage_cost(self):
        total_cost = 0
        if not self.operations:
            return total_cost
        sorted_ops = sorted(self.operations, key=lambda x: x['date'])
        current_storage = 0
        for i in range(len(sorted_ops) - 1):
            op = sorted_ops[i]
            next_op = sorted_ops[i + 1]
            if op['type'] == 'inject':
                current_storage += op['amount']
            else:
                current_storage -= op['amount']
            days = (next_op['date'] - op['date']).days
            storage_cost = current_storage * self.storage_cost_per_day * days
            total_cost += storage_cost
        return total_cost

    def calculate_value(self):
        total_revenue = sum(op['amount'] * op['price'] for op in self.operations if op['type'] == 'withdraw')
        total_cost = sum(op['amount'] * op['price'] for op in self.operations if op['type'] == 'inject')
        storage_cost = self.calculate_storage_cost()
        net_value = total_revenue - total_cost - storage_cost
        return {
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'storage_cost': storage_cost,
            'net_value': net_value
        }

# Example usage
contract = GasStorageContractEN(
    max_storage=100000,
    injection_rate=1000,
    withdrawal_rate=1500,
    storage_cost_per_day=0.01
)

# Perform injection and withdrawal
contract.inject_gas(50000, datetime(2025, 6, 1), 10)
contract.withdraw_gas(50000, datetime(2025, 12, 1), 12)

# Calculate and print contract value
value = contract.calculate_value()
print(f"Total Revenue: {value['total_revenue']}")
print(f"Total Cost: {value['total_cost']}")
print(f"Storage Cost: {value['storage_cost']}")
print(f"Net Profit: {value['net_value']}")

