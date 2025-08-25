"""
Example training script that WILL fail with OOM
Run preflight check to see the prediction!

This demonstrates how variable sequence lengths can cause memory overflow.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class TransformerModel(nn.Module):
    """A transformer model that will cause OOM"""
    
    def __init__(self):
        super().__init__()
        # Large transformer configuration
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2048,
                nhead=16,
                dim_feedforward=8192,
                batch_first=True
            ),
            num_layers=24
        )
        self.output = nn.Linear(2048, 10)
        
    def forward(self, x):
        x = self.transformer(x)
        return self.output(x.mean(dim=1))

class VariableLengthDataset(Dataset):
    """Dataset with variable sequence lengths - the killer!"""
    
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        # Sequence length increases after step 1000
        # This is what causes the OOM!
        if idx < 1000:
            seq_len = 512
        else:
            seq_len = 8192  # 16x longer - quadratic memory scaling!
        
        return torch.randn(seq_len, 2048), torch.randint(0, 10, (1,))

def train():
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Data setup
    dataset = VariableLengthDataset()
    dataloader = DataLoader(
        dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=0  # Single process - inefficient!
    )
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.squeeze().to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Step {batch_idx}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    train()