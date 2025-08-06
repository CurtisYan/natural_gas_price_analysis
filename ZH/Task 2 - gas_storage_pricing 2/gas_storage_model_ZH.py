"""
天然气存储合约定价模型
作者: Curtis Yan
日期: 2025.8.6

这个模型用于计算天然气存储合约的价值，包含注入、提取和存储成本的完整考虑。
"""

import numpy as np
from datetime import datetime, timedelta

class GasStorageContractCN:
    def __init__(self, max_storage, injection_rate, withdrawal_rate, storage_cost_per_day):
        self.max_storage = max_storage
        self.injection_rate = injection_rate
        self.withdrawal_rate = withdrawal_rate
        self.storage_cost_per_day = storage_cost_per_day
        self.current_storage = 0
        self.operations = []

    def inject_gas(self, amount, date, price):
        if self.current_storage + amount > self.max_storage:
            raise ValueError("超过最大存储容量")
        self.current_storage += amount
        self.operations.append({'date': date, 'amount': amount, 'price': price, 'type': 'inject'})
        return f"注入成功: {amount} 单位，价格: {price}"

    def withdraw_gas(self, amount, date, price):
        if self.current_storage < amount:
            raise ValueError("存储量不足")
        self.current_storage -= amount
        self.operations.append({'date': date, 'amount': amount, 'price': price, 'type': 'withdraw'})
        return f"提取成功: {amount} 单位，价格: {price}"

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

# 示例用法
contract = GasStorageContractCN(
    max_storage=100000,
    injection_rate=1000,
    withdrawal_rate=1500,
    storage_cost_per_day=0.01
)

# 注入和提取
contract.inject_gas(50000, datetime(2025, 6, 1), 10)
contract.withdraw_gas(50000, datetime(2025, 12, 1), 12)

# 计算并输出合约价值
value = contract.calculate_value()
print(f"总收入: {value['total_revenue']}")
print(f"总成本: {value['total_cost']}")
print(f"存储成本: {value['storage_cost']}")
print(f"净利润: {value['net_value']}")

