"""
Example demonstrating gradient explosion risks.
Deep model without proper stability measures.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class DeepUnstableModel(nn.Module):
    """Very deep model prone to gradient explosion"""
    
    def __init__(self):
        super().__init__()
        # Very deep transformer - 48 layers!
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=1024,
                nhead=16,
                dim_feedforward=4096,
                batch_first=True
            ),
            num_layers=48  # Very deep - gradient explosion risk!
        )
        self.output = nn.Linear(1024, 1000)
        
    def forward(self, x):
        x = self.transformer(x)
        return self.output(x.mean(dim=1))

class SimpleDataset(Dataset):
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        return torch.randn(512, 1024), torch.randint(0, 1000, (1,))

def train():
    model = DeepUnstableModel()
    
    # High learning rate for deep model - dangerous!
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # No gradient clipping - will explode!
    # No mixed precision scaling - additional risk!
    
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=64)
    
    criterion = nn.CrossEntropyLoss()
    
    # High gradient accumulation without clipping
    gradient_accumulation_steps = 32
    
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(dataloader):
            output = model(data)
            loss = criterion(output, target.squeeze())
            
            # Accumulate gradients - can cause overflow
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # No gradient clipping here - BOOM!
                optimizer.step()
                optimizer.zero_grad()

if __name__ == "__main__":
    train()