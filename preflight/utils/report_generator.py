"""
Generate reports in various formats
"""

import json
from typing import Any
from ..analyzer import PreflightResult

class ReportGenerator:
    """Generate analysis reports"""
    
    def to_json(self, result: PreflightResult) -> str:
        """
        Convert result to JSON format
        """
        report = {
            'memory_safe': result.memory_safe,
            'memory_peak_gb': result.memory_peak_gb,
            'failure_step': result.failure_step,
            'efficiency_score': result.efficiency_score,
            'warnings': result.warnings,
            'critical_errors': result.critical_errors,
            'breakdown': result.breakdown,
            'recommendations': result.recommendations
        }
        return json.dumps(report, indent=2)
    
    def to_html(self, result: PreflightResult) -> str:
        """
        Generate HTML report
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI Training Pre-Flight Check Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        .warning {{ color: orange; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Training Pre-Flight Check Report</h1>
    </div>
    
    <div class="section">
        <h2>Memory Analysis</h2>
        <p>Peak Usage: {result.memory_peak_gb:.1f} GB</p>
        <p>Status: <span class="{'pass' if result.memory_safe else 'fail'}">
            {'SAFE' if result.memory_safe else 'OVERFLOW PREDICTED'}
        </span></p>
    </div>
    
    <div class="section">
        <h2>Efficiency Score</h2>
        <p>{result.efficiency_score:.1%}</p>
    </div>
    
    <div class="section">
        <h2>Warnings</h2>
        <ul>
            {''.join(f'<li>{w}</li>' for w in result.warnings)}
        </ul>
    </div>
    
    <div class="section">
        <h2>Critical Errors</h2>
        <ul>
            {''.join(f'<li class="fail">{e}</li>' for e in result.critical_errors)}
        </ul>
    </div>
</body>
</html>
"""
        return html