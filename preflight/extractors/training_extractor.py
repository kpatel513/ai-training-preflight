"""
Extract training loop and configuration information
"""

import re
from typing import Dict, Any

class TrainingExtractor:
    """Extract training-related information"""
    
    def extract(self, script_content: str) -> Dict[str, Any]:
        """
        Extract training loop configuration
        """
        training_info = {
            'batch_size': 32,
            'gradient_accumulation': 1,
            'mixed_precision': False,
            'distributed': False,
            'num_gpus': 1,
            'training_steps': None,
            'num_epochs': None,
            'checkpoint_interval': None
        }
        
        # Extract batch size (if not found by DataExtractor)
        batch_pattern = r'batch_size[\s=]+(\d+)'
        match = re.search(batch_pattern, script_content)
        if match:
            training_info['batch_size'] = int(match.group(1))
        
        # Check for gradient accumulation
        if 'gradient_accumulation' in script_content or 'accumulation_steps' in script_content:
            accum_pattern = r'(?:gradient_accumulation|accumulation_steps)[\s=]+(\d+)'
            match = re.search(accum_pattern, script_content)
            if match:
                training_info['gradient_accumulation'] = int(match.group(1))
        
        # Check for mixed precision
        mixed_precision_indicators = ['autocast', 'fp16', 'amp', 'mixed_precision']
        if any(indicator in script_content for indicator in mixed_precision_indicators):
            training_info['mixed_precision'] = True
        
        # Check for distributed training
        distributed_indicators = [
            'DistributedDataParallel',
            'DDP',
            'DataParallel',
            'torch.distributed'
        ]
        if any(indicator in script_content for indicator in distributed_indicators):
            training_info['distributed'] = True
            
            # Try to find number of GPUs
            gpu_patterns = [
                r'world_size[\s=]+(\d+)',
                r'num_gpus[\s=]+(\d+)',
                r'n_gpus[\s=]+(\d+)'
            ]
            for pattern in gpu_patterns:
                match = re.search(pattern, script_content)
                if match:
                    training_info['num_gpus'] = int(match.group(1))
                    break
        
        # Extract training steps
        steps_pattern = r'(?:num_steps|training_steps|total_steps)[\s=]+(\d+)'
        match = re.search(steps_pattern, script_content)
        if match:
            training_info['training_steps'] = int(match.group(1))
        
        # Extract epochs
        epochs_pattern = r'(?:num_epochs|n_epochs|epochs)[\s=]+(\d+)'
        match = re.search(epochs_pattern, script_content)
        if match:
            training_info['num_epochs'] = int(match.group(1))
        
        return training_info