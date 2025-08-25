"""
Minimal analyzer stubs returning structured data expected by TrainingPreflightAnalyzer.
Replace with real implementations as needed.
"""

from typing import Dict, Any


class MemoryAnalyzer:
    def __init__(self, gpu_memory_gb: float):
        self.gpu_memory_gb = gpu_memory_gb

    def analyze(
        self,
        model_info: Dict[str, Any],
        data_info: Dict[str, Any],
        training_info: Dict[str, Any],
        optimizer_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        peak = min(self.gpu_memory_gb * 0.7, self.gpu_memory_gb)
        return {
            "will_oom": peak > self.gpu_memory_gb,
            "peak_memory_gb": peak,
            "failure_step": None,
            "warnings": [],
            "errors": [],
            "breakdown": {"model": "~ X GB", "optimizer": "~ Y GB", "total": peak},
            "recommendations": [],
        }


class EfficiencyAnalyzer:
    def analyze(
        self,
        model_info: Dict[str, Any],
        data_info: Dict[str, Any],
        training_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"score": 0.85, "warnings": [], "recommendations": []}


class StabilityAnalyzer:
    def analyze(
        self,
        model_info: Dict[str, Any],
        optimizer_info: Dict[str, Any],
        training_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"warnings": [], "errors": [], "recommendations": []}


