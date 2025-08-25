"""
Memory usage analysis and OOM prediction
"""

from typing import Dict, Any
from ..config import Config

class MemoryAnalyzer:
    """Analyze memory usage and predict OOM"""
    
    def __init__(self, gpu_memory_gb: float):
        self.gpu_memory_gb = gpu_memory_gb
        self.config = Config()
    
    def analyze(self, model_info: Dict, data_info: Dict, 
                training_info: Dict, optimizer_info: Dict) -> Dict[str, Any]:
        """
        Analyze memory requirements
        """
        result = {
            'peak_memory_gb': 0,
            'will_oom': False,
            'failure_step': None,
            'breakdown': {},
            'warnings': [],
            'recommendations': [],
            'errors': []
        }
        
        if not model_info.get('num_params'):
            result['warnings'].append("Cannot determine model size - memory analysis limited")
            return result
        
        # Calculate memory components
        memory_breakdown = self._calculate_memory_breakdown(
            model_info, data_info, training_info, optimizer_info
        )
        
        total_memory = sum(memory_breakdown.values())
        
        result['peak_memory_gb'] = total_memory
        result['will_oom'] = total_memory > self.gpu_memory_gb
        result['breakdown'] = {
            k: f"{v:.2f} GB" for k, v in memory_breakdown.items()
        }
        result['breakdown']['total'] = f"{total_memory:.2f} GB"
        
        # Add warnings and recommendations
        if total_memory > self.gpu_memory_gb * self.config.MEMORY_SAFETY_MARGIN:
            result['warnings'].append(
                f"Memory usage near limit: {total_memory:.1f}/{self.gpu_memory_gb} GB"
            )
        
        if memory_breakdown.get('activations', 0) > memory_breakdown.get('model', 0) * 5:
            result['recommendations'].append(
                "Activation memory is very high - consider gradient checkpointing"
            )
        
        if data_info.get('variable_length'):
            result['warnings'].append(
                "Variable sequence lengths detected - memory usage will fluctuate"
            )
        
        # Predict failure step for variable length sequences
        if data_info.get('variable_length') and total_memory > self.gpu_memory_gb:
            result['failure_step'] = self._predict_failure_step(
                data_info, memory_breakdown
            )
        
        return result
    
    def _calculate_memory_breakdown(self, model_info, data_info, 
                                   training_info, optimizer_info) -> Dict[str, float]:
        """Calculate memory for each component"""
        
        bytes_per_param = 2 if training_info.get('mixed_precision') else 4
        
        # Model parameters
        model_memory = model_info['num_params'] * bytes_per_param / 1e9
        
        # Optimizer states
        if optimizer_info.get('type') in ['adam', 'adamw']:
            optimizer_memory = model_memory * 2  # Two states per parameter
        elif optimizer_info.get('type') == 'sgd':
            optimizer_memory = model_memory if optimizer_info.get('momentum') else 0
        else:
            optimizer_memory = 0
        
        # Gradients
        gradient_memory = model_memory
        if training_info.get('gradient_accumulation', 1) > 1:
            gradient_memory *= training_info['gradient_accumulation']
        
        # Activations
        activation_memory = self._calculate_activation_memory(
            model_info, data_info, training_info, bytes_per_param
        )
        
        return {
            'model': model_memory,
            'optimizer': optimizer_memory,
            'gradients': gradient_memory,
            'activations': activation_memory
        }
    
    def _calculate_activation_memory(self, model_info, data_info, 
                                    training_info, bytes_per_param) -> float:
        """Calculate activation memory for forward pass"""
        
        if model_info['type'] != 'transformer':
            # Simple estimate for non-transformer models
            return model_info.get('num_params', 0) * bytes_per_param / 1e9 * 0.5
        
        # Get dimensions
        seq_len = data_info.get('max_sequence_length') or self.config.DEFAULT_SEQUENCE_LENGTH
        batch_size = training_info.get('batch_size') or self.config.DEFAULT_BATCH_SIZE
        hidden = model_info.get('hidden_size') or self.config.DEFAULT_HIDDEN_SIZE
        layers = model_info.get('num_layers') or self.config.DEFAULT_NUM_LAYERS
        heads = model_info.get('num_heads') or self.config.DEFAULT_NUM_HEADS
        
        # Attention matrices: seq_len × seq_len per head per layer
        # This is the quadratic scaling problem
        attention_memory = (
            seq_len * seq_len * layers * heads * batch_size * bytes_per_param
        ) / 1e9
        
        # FFN activations and layer outputs
        ffn_memory = (
            seq_len * batch_size * hidden * layers * 4 * bytes_per_param
        ) / 1e9
        
        # Total activation memory (attention + FFN + overhead)
        total_activation_memory = (attention_memory + ffn_memory) * 1.2  # 20% overhead
        
        return total_activation_memory
    
    def _predict_failure_step(self, data_info, memory_breakdown) -> int:
        """Predict at which step OOM will occur"""
        # Simplified prediction - in reality would analyze the training loop
        if data_info.get('variable_length'):
            # Assume failure happens when sequence length increases
            return 1001
        return None