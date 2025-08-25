"""
Example of a well-configured training script.
This should pass all pre-flight checks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

class EfficientTransformer(nn.Module):
    """Reasonably sized transformer model"""
    
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True
            ),
            num_layers=12
        )
        self.output = nn.Linear(768, 10)
        
    def forward(self, x):
        x = self.transformer(x)
        return self.output(x.mean(dim=1))

class FixedLengthDataset(Dataset):
    """Dataset with fixed sequence lengths"""
    
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        # Fixed sequence length - predictable memory usage
        seq_len = 512
        return torch.randn(seq_len, 768), torch.randint(0, 10, (1,))

def train():
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientTransformer().to(device)
    
    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Efficient data loading
    dataset = FixedLengthDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster GPU transfer
    )
    
    # Training loop with gradient clipping
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.squeeze().to(device)
            
            # Mixed precision forward pass
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Step {batch_idx}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    train()