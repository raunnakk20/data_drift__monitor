"""
baseline_calculator.py
Calculate baseline statistics from training data
Author: Your Name
Date: 2026-08-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

class BaselineCalculator:
    """Calculate and save baseline statistics from training data"""
    
    def __init__(self, data_path='data/training/fraud_train.csv'):
        self.data_path = data_path
        self.df = None
        self.stats = {}
        self.numerical_features = []
        self.categorical_features = []
        print(f"📊 Baseline Calculator initialized")
        print(f"📁 Data path: {data_path}")
    
    def load_data(self):
        """Load the CSV file into a pandas DataFrame"""
        print("\n📂 Loading data...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df)} transactions")
            print(f"   Columns: {self.df.columns.tolist()}")
            print(f"   Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print(f"❌ Error: File not found at {self.data_path}")
            print("   Run data_generator.py first to create the file!")
            return None
    
    def identify_feature_types(self):
        """Separate numerical and categorical features"""
        print("\n🔍 Identifying feature types...")
        
        exclude_cols = ['transaction_id', 'user_id', 'is_fraud']
        
        self.numerical_features = [
            col for col in self.df.select_dtypes(include=[np.number]).columns 
            if col not in exclude_cols
        ]
        
        self.categorical_features = [
            col for col in self.df.select_dtypes(include=['object']).columns 
            if col not in exclude_cols
        ]
        
        print(f"   Numerical features: {self.numerical_features}")
        print(f"   Categorical features: {self.categorical_features}")
        
        return self.numerical_features, self.categorical_features
    
    def calculate_numerical_stats(self):
        """Calculate statistics for numerical features"""
        print("\n📊 Calculating numerical statistics...")
        
        for feature in self.numerical_features:
            print(f"   Processing: {feature}...")
            
            series = self.df[feature].dropna()
            
            # Calculate quartiles
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            lower_fence = q25 - 1.5 * iqr
            upper_fence = q75 + 1.5 * iqr
            
            stats_dict = {
                'count': int(series.count()),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'mode': float(series.mode()[0]) if not series.mode().empty else None,
                'std': float(series.std()),
                'variance': float(series.var()),
                'min': float(series.min()),
                'max': float(series.max()),
                'q25': float(q25),
                'q50': float(series.quantile(0.50)),
                'q75': float(q75),
                'iqr': float(iqr),
                'lower_fence': float(lower_fence),
                'upper_fence': float(upper_fence),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'outlier_count': int(((series < lower_fence) | (series > upper_fence)).sum()),
                'outlier_percentage': float(((series < lower_fence) | (series > upper_fence)).mean() * 100)
            }
            
            self.stats[feature] = stats_dict
        
        print(f"✅ Calculated statistics for {len(self.numerical_features)} features")
        return self.stats
    
    def calculate_categorical_stats(self):
        """Calculate statistics for categorical features"""
        print("\n📊 Calculating categorical statistics...")
        
        for feature in self.categorical_features:
            print(f"   Processing: {feature}...")
            
            value_counts = self.df[feature].value_counts()
            
            stats_dict = {
                'unique_values': int(self.df[feature].nunique()),
                'most_common': value_counts.index[0] if not value_counts.empty else None,
                'most_common_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                'value_counts': value_counts.head(10).to_dict()
            }
            
            self.stats[feature] = stats_dict
        
        print(f"✅ Calculated statistics for {len(self.categorical_features)} features")
        return self.stats
    
    def print_summary(self):
        """Print a human-readable summary of statistics"""
        print("\n" + "="*60)
        print("📊 BASELINE STATISTICS SUMMARY")
        print("="*60)
        
        for feature, stats in self.stats.items():
            if feature in self.numerical_features:
                print(f"\n📈 {feature.upper()}:")
                print(f"   Count: {stats['count']:,}")
                print(f"   Mean: ${stats['mean']:,.2f}")
                print(f"   Median: ${stats['median']:,.2f}")
                print(f"   Mode: ${stats['mode']:,.2f}")
                print(f"   Std Dev: ${stats['std']:,.2f}")
                print(f"   Variance: ${stats['variance']:,.2f}")
                print(f"   Min: ${stats['min']:,.2f}")
                print(f"   Max: ${stats['max']:,.2f}")
                print(f"   Q25: ${stats['q25']:,.2f}")
                print(f"   Q75: ${stats['q75']:,.2f}")
                print(f"   IQR: ${stats['iqr']:,.2f}")
                print(f"   Skewness: {stats['skewness']:,.2f}")
                print(f"   Kurtosis: {stats['kurtosis']:,.2f}")
                print(f"   Outliers: {stats['outlier_count']:,} ({stats['outlier_percentage']:.1f}%)")
            
            elif feature in self.categorical_features:
                print(f"\n🏷️ {feature.upper()}:")
                print(f"   Unique values: {stats['unique_values']}")
                print(f"   Most common: {stats['most_common']} ({stats['most_common_count']:,})")
                print(f"   Top 5 values:")
                for value, count in list(stats['value_counts'].items())[:5]:
                    print(f"     - {value}: {count:,}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📈 Creating visualizations...")
        
        os.makedirs('visualizations', exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Distribution plots for numerical features
        for feature in self.numerical_features[:4]:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1 = axes[0]
            ax1.hist(self.df[feature], bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(self.df[feature].mean(), color='red', linestyle='--', label=f'Mean: {self.df[feature].mean():.2f}')
            ax1.axvline(self.df[feature].median(), color='green', linestyle='--', label=f'Median: {self.df[feature].median():.2f}')
            ax1.set_title(f'{feature.upper()} Distribution')
            ax1.set_xlabel(feature.capitalize())
            ax1.set_ylabel('Frequency')
            ax1.legend()
            
            ax2 = axes[1]
            ax2.boxplot(self.df[feature].dropna())
            ax2.set_title(f'{feature.upper()} Box Plot')
            ax2.set_ylabel(feature.capitalize())
            ax2.set_xticklabels([feature])
            
            plt.tight_layout()
            plt.savefig(f'visualizations/{feature}_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Created: {feature}_distribution.png")
        
        # 2. Correlation heatmap
        if len(self.numerical_features) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = self.df[self.numerical_features].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', square=True, linewidths=0.5, ax=ax)
            ax.set_title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig('visualizations/correlation_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("   ✅ Created: correlation_heatmap.png")
        
        # 3. Pair plot
        if len(self.numerical_features) >= 2:
            features_to_plot = self.numerical_features[:4]
            if len(features_to_plot) >= 2:
                fig = sns.pairplot(self.df[features_to_plot], diag_kind='kde')
                fig.savefig('visualizations/pairplot.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("   ✅ Created: pairplot.png")
        
        # 4. Categorical feature bar charts
        for feature in self.categorical_features[:2]:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_values = self.df[feature].value_counts().head(10)
            top_values.plot(kind='bar', ax=ax)
            ax.set_title(f'{feature.upper()} Distribution')
            ax.set_xlabel(feature.capitalize())
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(f'visualizations/{feature}_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Created: {feature}_distribution.png")
        
        # 5. Fraud vs Non-Fraud comparison
        if 'is_fraud' in self.df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1 = axes[0]
            fraud_data = [
                self.df[self.df['is_fraud'] == 0]['amount'],
                self.df[self.df['is_fraud'] == 1]['amount']
            ]
            bp = ax1.boxplot(fraud_data)
            ax1.set_xticklabels(['Normal', 'Fraud'])
            ax1.set_title('Amount by Fraud Status')
            ax1.set_ylabel('Transaction Amount')
            
            ax2 = axes[1]
            fraud_rate = self.df['is_fraud'].value_counts()
            ax2.pie(fraud_rate, labels=['Normal', 'Fraud'], autopct='%1.1f%%', 
                   colors=['green', 'red'])
            ax2.set_title('Fraud Rate')
            
            plt.tight_layout()
            plt.savefig('visualizations/fraud_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("   ✅ Created: fraud_analysis.png")
        
        print("✅ All visualizations complete!")
        print(f"📁 Visualizations saved to: visualizations/")
    
    def save_stats(self, filepath='baseline_stats.json'):
        """Save statistics to a JSON file"""
        print(f"\n💾 Saving statistics to: {filepath}")
        
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"✅ Statistics saved to: {filepath}")
        return filepath
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "="*60)
        print("🚀 STARTING BASELINE ANALYSIS")
        print("="*60)
        
        if self.load_data() is None:
            return None
        
        self.identify_feature_types()
        self.calculate_numerical_stats()
        self.calculate_categorical_stats()
        self.print_summary()
        self.create_visualizations()
        self.save_stats()
        
        print("\n" + "="*60)
        print("🎉 LEVEL 2 COMPLETE!")
        print("="*60)
        print("📊 Statistics saved to: baseline_stats.json")
        print("📈 Visualizations saved to: visualizations/")
        print("="*60)
        
        return self.stats

if __name__ == "__main__":
    print("🚀 Running Baseline Calculator...")
    print("="*60)
    
    calculator = BaselineCalculator()
    stats = calculator.run_full_analysis()
    
    print("\n" + "="*60)
    print("🎉 Level 2 Complete! Ready for Level 3.")
    print("="*60)