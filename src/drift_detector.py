"""
drift_detector.py - FINAL WORKING VERSION
Detect data drift between baseline and live data
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import os

class DriftDetector:
    def __init__(self):
        print("🔍 Drift Detector")
        self.baseline = None
        self.features = ['amount', 'hour', 'distance', 'frequency']
    
    def load_baseline(self):
        """Load baseline statistics"""
        try:
            with open('baseline_stats.json', 'r') as f:
                self.baseline = json.load(f)
            print("✅ Baseline loaded")
            return True
        except FileNotFoundError:
            print("❌ baseline_stats.json not found")
            print("   Run: python src/baseline_calculator.py")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def generate_live_data(self):
        """Generate live data with drift"""
        print("\n📊 Generating live data...")
        
        np.random.seed(42)
        n_samples = 1000
        data = {}
        
        for feature in self.features:
            if feature in self.baseline:
                mean = self.baseline[feature]['mean']
                std = self.baseline[feature]['std']
                # 20% drift
                data[feature] = np.random.normal(mean * 1.2, std * 1.1, n_samples)
        
        df = pd.DataFrame(data)
        df = df.clip(lower=0)
        
        os.makedirs('data/live', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data/live/live_batch_{timestamp}.csv"
        df.to_csv(filepath, index=False)
        
        print(f"✅ Generated {n_samples} transactions")
        return df
    
    def detect_drift(self, df):
        """Detect drift in live data"""
        print("\n🔍 Detecting drift...")
        results = {}
        
        for feature in self.features:
            if feature not in df.columns or feature not in self.baseline:
                continue
            
            baseline_mean = self.baseline[feature]['mean']
            baseline_std = self.baseline[feature]['std']
            live_values = df[feature].dropna()
            live_mean = live_values.mean()
            
            change = ((live_mean - baseline_mean) / abs(baseline_mean)) * 100
            
            np.random.seed(42)
            baseline_sample = np.random.normal(baseline_mean, baseline_std, len(live_values))
            ks_stat, p_value = stats.ks_2samp(baseline_sample, live_values)
            
            drift_detected = p_value < 0.05
            
            results[feature] = {
                'baseline_mean': float(baseline_mean),
                'live_mean': float(live_mean),
                'relative_change': float(change),
                'p_value': float(p_value),
                'drift_detected': bool(drift_detected)
            }
            
            status = "⚠️ DRIFT" if drift_detected else "✅ STABLE"
            print(f"   {feature:12} {status} (p={p_value:.4f}, change={change:+.1f}%)")
        
        drifted = [f for f, d in results.items() if d['drift_detected']]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'numerical_drift': results,
            'drifted_features': drifted,
            'total_features': len(results),
            'summary': f"{len(drifted)} out of {len(results)} features drifted",
            'drift_detected': len(drifted) > 0
        }
        
        # Save JSON - SIMPLE AND CLEAN
        with open('drift_results.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Saved: drift_results.json")
        
        return report

if __name__ == "__main__":
    print("="*60)
    print("🚀 DRIFT DETECTOR")
    print("="*60)
    
    detector = DriftDetector()
    
    if not detector.load_baseline():
        exit()
    
    live_df = detector.generate_live_data()
    report = detector.detect_drift(live_df)
    
    print("\n" + "="*60)
    print(f"📊 {report['summary']}")
    if report['drift_detected']:
        print("⚠️ DRIFT DETECTED!")
    print("="*60)
    print("\n🎉 LEVEL 3 COMPLETE!")