"""
Extract model architecture information from training scripts
"""

import ast
import re
from typing import Dict, Any

class ModelExtractor:
    """Extract model-related information"""
    
    def extract(self, script_content: str) -> Dict[str, Any]:
        """
        Extract model architecture details
        """
        try:
            tree = ast.parse(script_content)
        except:
            return self._default_model_info()
        
        model_info = {
            'type': 'unknown',
            'hidden_size': None,
            'num_layers': None,
            'num_heads': None,
            'num_params': None,
            'architecture': None
        }
        
        # Check for transformer variants
        if self._is_transformer(script_content):
            model_info['type'] = 'transformer'
            model_info.update(self._extract_transformer_params(script_content))
        elif self._is_cnn(script_content):
            model_info['type'] = 'cnn'
            model_info.update(self._extract_cnn_params(script_content))
        
        # Estimate parameters
        if model_info['hidden_size'] and model_info['num_layers']:
            model_info['num_params'] = self._estimate_parameters(model_info)
        
        return model_info
    
    def _default_model_info(self) -> Dict:
        """Return default model info when parsing fails"""
        return {
            'type': 'unknown',
            'hidden_size': None,
            'num_layers': None,
            'num_heads': None,
            'num_params': None,
            'architecture': None
        }
    
    def _is_transformer(self, content: str) -> bool:
        """Check if model is transformer-based"""
        transformer_keywords = [
            'Transformer', 'BERT', 'GPT', 'T5', 'RoBERTa',
            'attention', 'self_attention', 'multi_head'
        ]
        return any(keyword in content for keyword in transformer_keywords)
    
    def _is_cnn(self, content: str) -> bool:
        """Check if model is CNN-based"""
        cnn_keywords = ['Conv2d', 'ConvNet', 'ResNet', 'CNN']
        return any(keyword in content for keyword in cnn_keywords)
    
    def _extract_transformer_params(self, content: str) -> Dict:
        """Extract transformer-specific parameters"""
        params = {}
        
        # Hidden size
        hidden_pattern = r'(?:hidden_size|d_model|embed_dim)[\s=]+(\d+)'
        match = re.search(hidden_pattern, content)
        if match:
            params['hidden_size'] = int(match.group(1))
        
        # Number of layers
        layers_pattern = r'(?:num_layers|n_layers|num_encoder_layers)[\s=]+(\d+)'
        match = re.search(layers_pattern, content)
        if match:
            params['num_layers'] = int(match.group(1))
        
        # Number of attention heads
        heads_pattern = r'(?:num_heads|n_heads|nhead|num_attention_heads)[\s=]+(\d+)'
        match = re.search(heads_pattern, content)
        if match:
            params['num_heads'] = int(match.group(1))
        
        # Feedforward dimension
        ff_pattern = r'(?:dim_feedforward|ffn_dim|intermediate_size)[\s=]+(\d+)'
        match = re.search(ff_pattern, content)
        if match:
            params['feedforward_dim'] = int(match.group(1))
        
        return params
    
    def _extract_cnn_params(self, content: str) -> Dict:
        """Extract CNN-specific parameters"""
        params = {}
        
        # Look for common CNN patterns
        layers_pattern = r'num_layers[\s=]+(\d+)'
        match = re.search(layers_pattern, content)
        if match:
            params['num_layers'] = int(match.group(1))
        
        return params
    
    def _estimate_parameters(self, model_info: Dict) -> int:
        """Estimate total number of parameters"""
        if model_info['type'] == 'transformer':
            h = model_info['hidden_size']
            l = model_info['num_layers']
            # Approximate: embeddings + attention + FFN + layer norms
            # Each layer has roughly 12h² parameters
            return l * (12 * h * h + 13 * h)
        elif model_info['type'] == 'cnn':
            # Rough estimate for CNNs
            return 25_000_000  # Default ResNet-50 size
        return 0