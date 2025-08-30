"""
AI Training Pre-Flight Check
Predict training failures before they happen
"""

from .analyzer import TrainingPreflightAnalyzer, PreflightResult

__version__ = "0.1.0"
__all__ = ["TrainingPreflightAnalyzer", "PreflightResult", "analyze"]

def analyze(script_path: str, gpu_memory_gb: float = 40.0):
    """
    Quick analysis function
    
    Args:
        script_path: Path to training script
        gpu_memory_gb: Available GPU memory
        
    Returns:
        PreflightResult object
    """
    analyzer = TrainingPreflightAnalyzer(gpu_memory_gb)
    return analyzer.analyze(script_path)