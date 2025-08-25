"""
Configuration constants and defaults
"""

class Config:
    """Configuration settings"""
    
    # Memory settings
    DEFAULT_GPU_MEMORY_GB = 40.0  # A100
    MEMORY_SAFETY_MARGIN = 0.9
    
    # Model defaults
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_SEQUENCE_LENGTH = 512
    DEFAULT_HIDDEN_SIZE = 768
    DEFAULT_NUM_LAYERS = 12
    DEFAULT_NUM_HEADS = 12
    
    # Efficiency thresholds
    EFFICIENCY_GOOD = 0.8
    EFFICIENCY_WARNING = 0.6
    EFFICIENCY_CRITICAL = 0.4
    
    # Stability settings
    MAX_GRADIENT_NORM = 1.0
    FP16_MAX = 65504
    FP32_MAX = 3.4e38
    
    # Analysis settings
    MAX_WARNINGS = 10
    MAX_RECOMMENDATIONS = 5