"""
ai_reporter.py - FIXED VERSION
Generate AI-powered drift report (No emojis)
"""

import json
import os
from datetime import datetime

class AIReporter:
    def __init__(self):
        print("AI Reporter (Mock Mode)")
    
    def run(self):
        print("\n" + "="*60)
        print("Loading drift_results.json...")
        print("="*60)
        
        # Check if file exists
        if not os.path.exists('drift_results.json'):
            print("ERROR: File not found!")
            print("   Run: python src/drift_detector.py")
            return
        
        # Read file
        try:
            with open('drift_results.json', 'r') as f:
                data = json.load(f)
            print("Data loaded successfully")
        except json.JSONDecodeError:
            print("ERROR: Corrupted file - deleting...")
            os.remove('drift_results.json')
            print("   Run drift_detector.py again!")
            return
        except Exception as e:
            print(f"ERROR: {e}")
            return
        
        # Generate HTML report
        print("\nGenerating HTML report...")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Data Drift Report</title>
    <style>
        body {{ font-family: Arial; margin: 40px; background: #f0f0f0; }}
        .container {{ max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        .alert {{ background: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .feature {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; background: #fafafa; }}
        .badge {{ background: #e74c3c; color: white; padding: 3px 10px; border-radius: 15px; }}
        .stable {{ background: #27ae60; color: white; padding: 3px 10px; border-radius: 15px; }}
        .footer {{ margin-top: 30px; color: #7f8c8d; font-size: 12px; border-top: 1px solid #ddd; padding-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Data Drift Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Status:</strong> {'DRIFT DETECTED' if data.get('drift_detected') else 'ALL STABLE'}</p>
        
        <div class="alert">
            <h2>Summary</h2>
            <p>{data.get('summary', 'Unknown')}</p>
        </div>
        
        <h2>Feature Analysis</h2>
"""
        
        for feature, stats in data.get('numerical_drift', {}).items():
            detected = stats.get('drift_detected', False)
            badge = "badge" if detected else "stable"
            status = "DRIFT" if detected else "Stable"
            html += f"""
        <div class="feature">
            <h3>{feature.upper()} <span class="{badge}">{status}</span></h3>
            <p><strong>Change:</strong> {stats.get('relative_change', 0):+.1f}%</p>
            <p><strong>P-Value:</strong> {stats.get('p_value', 1):.4f}</p>
            <p><strong>Baseline:</strong> {stats.get('baseline_mean', 0):.2f} -> <strong>Current:</strong> {stats.get('live_mean', 0):.2f}</p>
        </div>
"""
        
        html += """
        <div class="footer">
            <p>Data Drift Monitor | AI Reporter</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open('drift_report.html', 'w', encoding='utf-8') as f:
            f.write(html)
        
        print("Report saved: drift_report.html")
        print("\n" + "="*60)
        print("LEVEL 4 COMPLETE!")
        print("="*60)
        print("Open drift_report.html in your browser")

if __name__ == "__main__":
    print("AI-Powered Drift Reporter")
    print("="*60)
    reporter = AIReporter()
    reporter.run()