"""
Main analyzer orchestrator
"""

import ast
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

from .config import Config
from .extractors import (
    ModelExtractor, 
    DataExtractor, 
    TrainingExtractor, 
    OptimizerExtractor
)
from .analyzers import (
    MemoryAnalyzer,
    EfficiencyAnalyzer,
    StabilityAnalyzer
)

@dataclass
class PreflightResult:
    """Results from pre-flight analysis"""
    memory_safe: bool
    memory_peak_gb: float
    failure_step: Optional[int]
    efficiency_score: float
    warnings: List[str]
    critical_errors: List[str]
    breakdown: Dict[str, Any]
    recommendations: List[str]

class TrainingPreflightAnalyzer:
    """
    Main analyzer that orchestrates all checks
    """
    
    def __init__(self, gpu_memory_gb: float = 40.0):
        self.gpu_memory_gb = gpu_memory_gb
        self.config = Config()
        
        # Initialize extractors
        self.model_extractor = ModelExtractor()
        self.data_extractor = DataExtractor()
        self.training_extractor = TrainingExtractor()
        self.optimizer_extractor = OptimizerExtractor()
        
        # Initialize analyzers
        self.memory_analyzer = MemoryAnalyzer(gpu_memory_gb)
        self.efficiency_analyzer = EfficiencyAnalyzer()
        self.stability_analyzer = StabilityAnalyzer()
    
    def analyze(self, script_path: str) -> PreflightResult:
        """
        Run complete analysis on a training script
        """
        try:
            # Read script
            script_content = Path(script_path).read_text()
        except Exception as e:
            return PreflightResult(
                memory_safe=False,
                memory_peak_gb=0,
                failure_step=None,
                efficiency_score=0,
                warnings=[],
                critical_errors=[f"Failed to read script: {e}"],
                breakdown={},
                recommendations=[]
            )
        
        # Extract information
        model_info = self.model_extractor.extract(script_content)
        data_info = self.data_extractor.extract(script_content)
        training_info = self.training_extractor.extract(script_content)
        optimizer_info = self.optimizer_extractor.extract(script_content)
        
        # Run analyses
        memory_result = self.memory_analyzer.analyze(
            model_info, data_info, training_info, optimizer_info
        )
        
        efficiency_result = self.efficiency_analyzer.analyze(
            model_info, data_info, training_info
        )
        
        stability_result = self.stability_analyzer.analyze(
            model_info, optimizer_info, training_info
        )
        
        # Compile results
        return self._compile_results(
            memory_result, efficiency_result, stability_result
        )
    
    def _compile_results(self, memory, efficiency, stability) -> PreflightResult:
        """Combine all analysis results"""
        
        warnings = []
        warnings.extend(memory.get('warnings', []))
        warnings.extend(efficiency.get('warnings', []))
        warnings.extend(stability.get('warnings', []))
        
        critical_errors = []
        if memory.get('will_oom'):
            critical_errors.append(
                f"MEMORY OVERFLOW: {memory['peak_memory_gb']:.1f}GB exceeds {self.gpu_memory_gb}GB"
            )
            if memory.get('failure_step'):
                critical_errors.append(f"Predicted OOM at step {memory['failure_step']}")
        
        critical_errors.extend(memory.get('errors', []))
        critical_errors.extend(stability.get('errors', []))
        
        recommendations = []
        recommendations.extend(memory.get('recommendations', []))
        recommendations.extend(efficiency.get('recommendations', []))
        recommendations.extend(stability.get('recommendations', []))
        
        return PreflightResult(
            memory_safe=not memory.get('will_oom', False),
            memory_peak_gb=memory.get('peak_memory_gb', 0),
            failure_step=memory.get('failure_step'),
            efficiency_score=efficiency.get('score', 1.0),
            warnings=warnings[:10],  # Limit warnings
            critical_errors=critical_errors,
            breakdown=memory.get('breakdown', {}),
            recommendations=recommendations[:5]  # Limit recommendations
        )
    
    def print_results(self, result: PreflightResult):
        """
        Pretty print the analysis results
        """
        print("\n" + "="*70)
        print("                    AI TRAINING PRE-FLIGHT CHECK")
        print("="*70)
        
        # Memory Analysis
        print("\nMEMORY ANALYSIS:")
        print(f"   Peak Usage: {result.memory_peak_gb:.1f}GB / {self.gpu_memory_gb}GB")
        if result.breakdown:
            print("   Breakdown:")
            for component, size in result.breakdown.items():
                if component != 'total':
                    print(f"     - {component}: {size}")
        
        if result.memory_safe:
            print(f"   Status: SAFE")
        else:
            print(f"   Status: OVERFLOW PREDICTED")
            if result.failure_step:
                print(f"   Will fail at step {result.failure_step}")
        
        # Efficiency Score
        print(f"\nEFFICIENCY ANALYSIS:")
        print(f"   Score: {result.efficiency_score:.1%}")
        if result.efficiency_score >= 0.8:
            print("   Status: Good efficiency")
        elif result.efficiency_score >= 0.6:
            print("   Status: Moderate efficiency issues")
        else:
            print("   Status: Severe efficiency problems")
        
        # Warnings
        if result.warnings:
            print("\nWARNINGS:")
            for i, warning in enumerate(result.warnings, 1):
                print(f"   {i}. {warning}")
        
        # Critical Errors
        if result.critical_errors:
            print("\nCRITICAL ERRORS:")
            for i, error in enumerate(result.critical_errors, 1):
                print(f"   {i}. {error}")
        
        # Recommendations
        if result.recommendations:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Final Decision
        print("\n" + "-"*70)
        if result.critical_errors:
            print("NO GO - Fix critical errors before training")
        elif result.warnings and result.efficiency_score < 0.5:
            print("PROCEED WITH CAUTION - Address warnings for better performance")
        else:
            print("CLEARED FOR TRAINING - All systems go")
        print("="*70 + "\n")