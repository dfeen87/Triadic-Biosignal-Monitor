"""
Performance Report Generator for Triadic Biosignal Monitor

Generates comprehensive HTML/PDF reports from validation results.

Usage:
    python generate_report.py --results results/validation_*.json --output report.html
    python generate_report.py --results results/ --format html --output report.html

Authors: Marcel Krüger, Don Feeney
Date: January 27, 2026
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
from datetime import datetime

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.metrics import format_performance_report, format_ablation_report


def generate_html_report(results: Dict, output_path: str) -> None:
    """Generate HTML report from validation results."""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Triadic Biosignal Monitor - Validation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 40px;
            background-color: #f5f7fa;
            color: #2c3e50;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 10px;
            font-size: 32px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 24px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #555;
            margin-top: 25px;
            font-size: 18px;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 14px;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric {{
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
            text-align: center;
            color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric.success {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .metric.warning {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .metric-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.9;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 1px;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .confusion-matrix {{
            display: grid;
            grid-template-columns: 150px 1fr 1fr;
            gap: 2px;
            margin: 20px 0;
            background-color: #ecf0f1;
        }}
        .cm-cell {{
            background-color: white;
            padding: 20px;
            text-align: center;
        }}
        .cm-header {{
            background-color: #3498db;
            color: white;
            padding: 15px;
            text-align: center;
            font-weight: bold;
        }}
        .cm-label {{
            background-color: #34495e;
            color: white;
            padding: 15px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .cm-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .good {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .bad {{ color: #e74c3c; }}
        .alert-summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .badge-success {{ background-color: #27ae60; color: white; }}
        .badge-warning {{ background-color: #f39c12; color: white; }}
        .badge-info {{ background-color: #3498db; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Triadic Biosignal Monitor</h1>
        <h2 style="margin-top: 0; border: none; font-size: 20px; color: #7f8c8d;">Validation Report</h2>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>⚙️ Configuration</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Processing Mode</td><td><strong>{results.get('mode', 'N/A').upper()}</strong></td></tr>
                <tr><td>Sampling Rate</td><td>{results.get('config', {}).get('fs', 'N/A')} Hz</td></tr>
                <tr><td>α (Spectral/Morphological)</td><td>{results.get('config', {}).get('alpha', 'N/A')}</td></tr>
                <tr><td>β (Information/Entropy)</td><td>{results.get('config', {}).get('beta', 'N/A')}</td></tr>
                <tr><td>γ (Coupling/Coherence)</td><td>{results.get('config', {}).get('gamma', 'N/A')}</td></tr>
                <tr><td>Threshold τ</td><td>{results.get('config', {}).get('threshold', 'N/A')}</td></tr>
            </table>
        </div>
"""
    
    # Add performance metrics if available
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        
        html += """
        <div class="section">
            <h2>📊 Performance Metrics</h2>
            <div class="metrics-grid">
"""
        
        # Key metrics with styling
        key_metrics = [
            ('sensitivity', 'Sensitivity', 'success'),
            ('specificity', 'Specificity', 'success'),
            ('precision', 'Precision', 'success'),
            ('f1_score', 'F1 Score', 'success'),
            ('false_alarm_rate', 'False Alarm Rate', 'warning')
        ]
        
        for metric_key, label, style in key_metrics:
            if metric_key in metrics:
                value = metrics[metric_key]
                html += f"""
                <div class="metric {style}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.3f if isinstance(value, float) else value}</div>
                </div>
"""
        
        html += """
            </div>
            
            <h3>Confusion Matrix</h3>
            <div class="confusion-matrix">
                <div class="cm-cell"></div>
                <div class="cm-header">Predicted Positive</div>
                <div class="cm-header">Predicted Negative</div>
                
                <div class="cm-label">Actual Positive</div>
                <div class="cm-cell"><div class="cm-value">{}</div><div>True Positives</div></div>
                <div class="cm-cell"><div class="cm-value">{}</div><div>False Negatives</div></div>
                
                <div class="cm-label">Actual Negative</div>
                <div class="cm-cell"><div class="cm-value">{}</div><div>False Positives</div></div>
                <div class="cm-cell"><div class="cm-value">{}</div><div>True Negatives</div></div>
            </div>
""".format(
            metrics.get('true_positives', 0),
            metrics.get('false_negatives', 0),
            metrics.get('false_positives', 0),
            metrics.get('true_negatives', 0)
        )
        
        html += """
        </div>
"""
    
    # Add alert summary
    if 'alerts' in results:
        n_alerts = len(results['alerts'])
        html += f"""
        <div class="section">
            <h2>🚨 Alert Summary</h2>
            <div class="alert-summary">
                <p style="font-size: 18px; margin: 0;">
                    Total alerts generated: <strong style="font-size: 32px; color: #3498db;">{n_alerts}</strong>
                </p>
            </div>
        </div>
"""
    
    # Add processing info if available
    if 'processing_time' in results:
        html += f"""
        <div class="section">
            <h2>⏱️ Processing Information</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Processing Time</td><td>{results['processing_time']:.2f} seconds</td></tr>
"""
        if 'timestamps' in results:
            html += f"""
                <tr><td>Windows Processed</td><td>{len(results['timestamps'])}</td></tr>
"""
        html += """
            </table>
        </div>
"""
    
    # Footer
    html += """
        <div style="margin-top: 60px; padding-top: 20px; border-top: 1px solid #ecf0f1; text-align: center; color: #95a5a6;">
            <p>Generated by Triadic Biosignal Monitor</p>
            <p style="font-size: 12px;">Authors: Marcel Krüger, Don Feeney</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✅ HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate validation report for Triadic Biosignal Monitor'
    )
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file or directory')
    parser.add_argument('--output', type=str, default='report.html',
                       help='Output file path (default: report.html)')
    parser.add_argument('--format', type=str, choices=['html', 'pdf'], default='html',
                       help='Output format (default: html)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Triadic Biosignal Monitor - Report Generator")
    print("="*60)
    
    # Load results
    results_path = Path(args.results)
    
    if results_path.is_file():
        print(f"\nLoading results from: {results_path}")
        with open(results_path, 'r') as f:
            results = json.load(f)
    elif results_path.is_dir():
        # Find most recent JSON file
        json_files = sorted(
            results_path.glob('*.json'), 
            key=lambda p: p.stat().st_mtime, 
            reverse=True
        )
        if not json_files:
            print(f"❌ No JSON files found in {results_path}")
            return 1
        print(f"\nLoading most recent results: {json_files[0]}")
        with open(json_files[0], 'r') as f:
            results = json.load(f)
    else:
        print(f"❌ Invalid results path: {results_path}")
        return 1
    
    # Generate report
    print(f"\nGenerating {args.format.upper()} report...")
    
    if args.format == 'html':
        generate_html_report(results, args.output)
    elif args.format == 'pdf':
        print("⚠️  PDF generation requires additional dependencies (reportlab)")
        print("Generating HTML version instead...")
        html_output = args.output.replace('.pdf', '.html')
        generate_html_report(results, html_output)
        print("\nTo convert HTML to PDF, you can:")
        print("  1. Open the HTML file in a browser and print to PDF")
        print("  2. Install wkhtmltopdf: https://wkhtmltopdf.org/")
        print("  3. Use: wkhtmltopdf report.html report.pdf")
    
    print("\n" + "="*60)
    print("✅ Report generation completed successfully!")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
