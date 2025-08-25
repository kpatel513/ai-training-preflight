"""
Extract data pipeline information
"""

import re
from typing import Dict, Any

class DataExtractor:
    """Extract data-related information"""
    
    def extract(self, script_content: str) -> Dict[str, Any]:
        """
        Extract data loader and dataset information
        """
        data_info = {
            'batch_size': self._extract_batch_size(script_content),
            'sequence_length': None,
            'max_sequence_length': None,
            'variable_length': False,
            'num_workers': 0,
            'dataset_size': None
        }
        
        # Extract sequence lengths
        seq_info = self._extract_sequence_info(script_content)
        data_info.update(seq_info)
        
        # Check for variable length sequences
        if self._has_variable_sequences(script_content):
            data_info['variable_length'] = True
        
        # Extract DataLoader workers
        workers_pattern = r'num_workers[\s=]+(\d+)'
        match = re.search(workers_pattern, script_content)
        if match:
            data_info['num_workers'] = int(match.group(1))
        
        return data_info
    
    def _extract_batch_size(self, content: str) -> int:
        """Extract batch size from script"""
        # Look for batch_size in various contexts
        patterns = [
            r'batch_size[\s=]+(\d+)',
            r'batch[\s=]+(\d+)',
            r'bs[\s=]+(\d+)',
            r'BATCH_SIZE[\s=]+(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return int(match.group(1))
        
        return 32  # Default
    
    def _extract_sequence_info(self, content: str) -> Dict:
        """Extract sequence length information"""
        info = {}
        
        # Look for sequence length patterns
        patterns = [
            r'seq_len[\s=]+(\d+)',
            r'sequence_length[\s=]+(\d+)',
            r'max_len[\s=]+(\d+)',
            r'max_length[\s=]+(\d+)',
            r'MAX_LEN[\s=]+(\d+)'
        ]
        
        lengths = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                lengths.append(int(match))
        
        if lengths:
            info['sequence_length'] = min(lengths)
            info['max_sequence_length'] = max(lengths)
        
        return info
    
    def _has_variable_sequences(self, content: str) -> bool:
        """Check if script uses variable length sequences"""
        indicators = [
            'if batch_idx',
            'if step',
            'if epoch',
            'variable_length',
            'dynamic_length'
        ]
        
        # Check for conditional sequence length changes
        for indicator in indicators:
            if indicator in content and 'seq_len' in content:
                return True
        
        # Check for multiple different sequence lengths
        seq_pattern = r'seq_len[\s=]+(\d+)'
        matches = re.findall(seq_pattern, content)
        if len(set(matches)) > 1:
            return True
        
        return False