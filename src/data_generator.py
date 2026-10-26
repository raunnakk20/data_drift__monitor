"""
data_generator.py
Generates synthetic transaction data for fraud detection
"""

import random
import uuid
import os
import pandas as pd

class TransactionGenerator:
    """Generates realistic transaction data with optional drift"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        print(f"🔧 Generator initialized with seed: {seed}")
    
    def generate_single_transaction(self):
        """Generate ONE transaction as a dictionary"""
        transaction_id = str(uuid.uuid4())
        user_id = random.randint(1000, 9999)
        amount = round(random.expovariate(1/100), 2)
        amount = max(5, min(5000, amount))
        hour = random.randint(0, 23)
        category = random.choice(['retail', 'grocery', 'entertainment', 'travel', 'restaurant'])
        distance = round(random.expovariate(1/20), 1)
        distance = max(0.1, min(200, distance))
        frequency = random.randint(1, 50)
        is_fraud = 1 if random.random() < 0.03 else 0
        
        return {
            'transaction_id': transaction_id,
            'user_id': user_id,
            'amount': amount,
            'hour': hour,
            'category': category,
            'distance': distance,
            'frequency': frequency,
            'is_fraud': is_fraud
        }
    
    def generate_dataset(self, n_samples=50000, drift=False):
        """Generate dataset with n_samples transactions"""
        transactions = []
        for _ in range(n_samples):
            transaction = self.generate_single_transaction()
            if drift:
                transaction['amount'] = round(transaction['amount'] * 1.5, 2)
                transaction['distance'] = round(transaction['distance'] * 2.0, 1)
                transaction['frequency'] = transaction['frequency'] + 15
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        
        current_fraud = df['is_fraud'].mean()
        if abs(current_fraud - 0.03) > 0.005:
            fraud_indices = df[df['is_fraud'] == 1].index.tolist()
            non_fraud_indices = df[df['is_fraud'] == 0].index.tolist()
            target_count = int(n_samples * 0.03)
            current_count = len(fraud_indices)
            
            if current_count < target_count:
                need = target_count - current_count
                convert = random.sample(non_fraud_indices, min(need, len(non_fraud_indices)))
                df.loc[convert, 'is_fraud'] = 1
            elif current_count > target_count:
                need = current_count - target_count
                convert = random.sample(fraud_indices, min(need, len(fraud_indices)))
                df.loc[convert, 'is_fraud'] = 0
        
        return df

if __name__ == "__main__":
    print("🚀 Testing Transaction Generator...")
    print("="*60)
    
    gen = TransactionGenerator(seed=42)
    
    print("\n📝 Single transaction example:")
    single = gen.generate_single_transaction()
    for key, value in single.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    
    print("\n📊 Generating 50,000 training transactions...")
    df_train = gen.generate_dataset(n_samples=50000, drift=False)
    
    os.makedirs('data/training', exist_ok=True)
    df_train.to_csv('data/training/fraud_train.csv', index=False)
    
    print(f"✅ Generated {len(df_train)} transactions")
    print(f"📁 Saved to: data/training/fraud_train.csv")
    print(f"\n📊 Summary:")
    print(f"  - Avg amount: ${df_train['amount'].mean():.2f}")
    print(f"  - Max amount: ${df_train['amount'].max():.2f}")
    print(f"  - Fraud rate: {df_train['is_fraud'].mean():.2%}")
    print(f"  - Top category: {df_train['category'].mode()[0]}")
    
    print("\n📊 Generating 1,000 drift transactions...")
    df_drift = gen.generate_dataset(n_samples=1000, drift=True)
    print(f"  - Avg amount (drift): ${df_drift['amount'].mean():.2f}")
    print(f"  - Avg amount (normal): ${df_train['amount'].mean():.2f}")
    print(f"  - Change: +{((df_drift['amount'].mean() / df_train['amount'].mean()) - 1) * 100:.1f}%")
    
    print("\n" + "="*60)
    print("🎉 Level 1 Complete! Ready for Level 2.")