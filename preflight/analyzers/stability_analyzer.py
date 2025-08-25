"""
Numerical stability analysis
"""

from typing import Dict, Any
from ..config import Config

class StabilityAnalyzer:
    """Analyze numerical stability risks"""
    
    def __init__(self):
        self.config = Config()
    
    def analyze(self, model_info: Dict, optimizer_info: Dict,
                training_info: Dict) -> Dict[str, Any]:
        """
        Analyze numerical stability
        """
        result = {
            'stable': True,
            'risks': [],
            'recommendations': [],
            'warnings': [],
            'errors': []
        }
        
        # Check for gradient explosion risk
        if not optimizer_info.get('gradient_clipping'):
            if model_info.get('num_layers') and model_info['num_layers'] > 12:
                result['warnings'].append(
                    "Deep model without gradient clipping - explosion risk"
                )
                result['recommendations'].append(
                    "Add gradient clipping with max_norm=1.0"
                )
        
        # Check for mixed precision overflow risk
        if training_info.get('mixed_precision'):
            if training_info.get('gradient_accumulation', 1) > 8:
                result['warnings'].append(
                    "High gradient accumulation with FP16 - potential overflow"
                )
                result['recommendations'].append(
                    "Monitor gradient scales or reduce accumulation steps"
                )
        
        # Check for learning rate issues
        lr = optimizer_info.get('learning_rate', 1e-4)
        if model_info.get('num_params'):
            if model_info['num_params'] > 1e9 and lr > 1e-3:
                result['warnings'].append(
                    f"Learning rate {lr} may be too high for large model"
                )
                result['recommendations'].append(
                    "Consider reducing learning rate to 1e-4 or lower"
                )
            elif model_info['num_params'] < 1e6 and lr < 1e-5:
                result['warnings'].append(
                    f"Learning rate {lr} may be too low for small model"
                )
        
        # Check for potential NaN issues
        if training_info.get('mixed_precision') and not optimizer_info.get('gradient_clipping'):
            result['warnings'].append(
                "Mixed precision without gradient clipping - NaN risk"
            )
            result['errors'].append(
                "High risk of NaN loss - add gradient clipping"
            )
            result['stable'] = False
        
        return result