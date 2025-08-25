"""
Extract optimizer configuration
"""

import re
from typing import Dict, Any

class OptimizerExtractor:
    """Extract optimizer-related information"""
    
    def extract(self, script_content: str) -> Dict[str, Any]:
        """
        Extract optimizer configuration
        """
        optimizer_info = {
            'type': 'adam',
            'learning_rate': 1e-4,
            'weight_decay': 0,
            'gradient_clipping': None,
            'warmup_steps': 0,
            'scheduler': None
        }
        
        # Identify optimizer type
        if 'AdamW' in script_content:
            optimizer_info['type'] = 'adamw'
        elif 'SGD' in script_content:
            optimizer_info['type'] = 'sgd'
        elif 'Adam' in script_content:
            optimizer_info['type'] = 'adam'
        elif 'RMSprop' in script_content:
            optimizer_info['type'] = 'rmsprop'
        
        # Extract learning rate
        lr_patterns = [
            r'learning_rate[\s=]+([0-9.e\-]+)',
            r'lr[\s=]+([0-9.e\-]+)',
            r'LR[\s=]+([0-9.e\-]+)'
        ]
        for pattern in lr_patterns:
            match = re.search(pattern, script_content)
            if match:
                try:
                    optimizer_info['learning_rate'] = float(match.group(1))
                    break
                except:
                    pass
        
        # Extract weight decay
        wd_pattern = r'weight_decay[\s=]+([0-9.e\-]+)'
        match = re.search(wd_pattern, script_content)
        if match:
            try:
                optimizer_info['weight_decay'] = float(match.group(1))
            except:
                pass
        
        # Check for gradient clipping
        if 'clip_grad' in script_content or 'grad_clip' in script_content:
            clip_patterns = [
                r'(?:clip_grad_norm|grad_clip|max_norm)[\s=]+([0-9.]+)',
                r'torch\.nn\.utils\.clip_grad_norm_.*max_norm[\s=]+([0-9.]+)'
            ]
            for pattern in clip_patterns:
                match = re.search(pattern, script_content)
                if match:
                    optimizer_info['gradient_clipping'] = float(match.group(1))
                    break
        
        # Check for scheduler
        scheduler_types = [
            'CosineAnnealingLR',
            'StepLR',
            'ExponentialLR',
            'ReduceLROnPlateau',
            'LinearLR',
            'OneCycleLR'
        ]
        for scheduler in scheduler_types:
            if scheduler in script_content:
                optimizer_info['scheduler'] = scheduler.lower()
                break
        
        return optimizer_info