"""
Command-line interface for preflight checks
"""

import click
import sys
from pathlib import Path
from .analyzer import TrainingPreflightAnalyzer

@click.group()
def cli():
    """AI Training Pre-Flight Check - Predict failures before they happen"""
    pass

@cli.command()
@click.argument('script_path', type=click.Path(exists=True))
@click.option('--gpu-memory', default=40.0, help='GPU memory in GB')
@click.option('--output', '-o', help='Output report file')
@click.option('--format', type=click.Choice(['terminal', 'json', 'html']), 
              default='terminal', help='Output format')
def check(script_path, gpu_memory, output, format):
    """
    Analyze a training script for potential failures
    """
    click.echo(f"\nAnalyzing: {script_path}\n")
    
    analyzer = TrainingPreflightAnalyzer(gpu_memory)
    result = analyzer.analyze(script_path)
    
    if format == 'terminal':
        analyzer.print_results(result)
    elif format == 'json':
        import json
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
        report_json = json.dumps(report, indent=2)
        if output:
            Path(output).write_text(report_json)
        else:
            click.echo(report_json)
    
    # Exit with error code if critical issues found
    if result.critical_errors:
        sys.exit(1)

@cli.command()
@click.argument('script_path', type=click.Path(exists=True))
def quick(script_path):
    """
    Quick check with summary only
    """
    analyzer = TrainingPreflightAnalyzer()
    result = analyzer.analyze(script_path)
    
    if result.critical_errors:
        click.echo("FAILED - Critical errors found")
        for error in result.critical_errors:
            click.echo(f"  • {error}")
    elif result.warnings:
        click.echo("PASSED WITH WARNINGS")
        click.echo(f"  Efficiency: {result.efficiency_score:.0%}")
    else:
        click.echo("PASSED - Ready for training")

def main():
    cli()

if __name__ == "__main__":
    main()