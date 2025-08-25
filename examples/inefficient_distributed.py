"""
Example showing distributed training inefficiencies.
Variable sequence lengths cause severe slowdowns in distributed training.
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

class SimpleTransformer(nn.Module):
    """Basic transformer model"""
    
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=1024,
                nhead=8,
                dim_feedforward=4096,
                batch_first=True
            ),
            num_layers=12
        )
        
    def forward(self, x):
        return self.transformer(x)

class ImbalancedDataset(Dataset):
    """Dataset with varying sequence lengths per sample"""
    
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        # Different samples have very different lengths
        # This causes severe imbalance in distributed training
        if idx % 10 == 0:
            seq_len = 2048  # Long sequences
        else:
            seq_len = 128   # Short sequences
        
        return torch.randn(seq_len, 1024)

def train():
    # This would be initialized with proper distributed setup
    world_size = 8  # 8 GPUs
    
    model = SimpleTransformer()
    # model = DDP(model)  # Would wrap in DDP
    
    # Poor configuration for distributed training
    dataset = ImbalancedDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=16,  # Too small for 8 GPUs
        shuffle=True,    # Should use DistributedSampler
        num_workers=0    # No parallel data loading
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # No gradient clipping
    
    # Training loop (simplified)
    for epoch in range(10):
        for batch in dataloader:
            # Variable length sequences cause stragglers
            # GPUs with short sequences wait for GPUs with long sequences
            output = model(batch)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

if __name__ == "__main__":
    train()