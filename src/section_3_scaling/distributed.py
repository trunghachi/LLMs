import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import os

# Sample dataset for training (placeholder)
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = [f"Sample text {i}" for i in range(size)]
        self.labels = [i % 2 for i in range(size)]  # Dummy labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Simple model for distributed training
class SimpleModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_size=128):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, 2)  # Binary classification
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Simple pooling
        return self.fc(x)

def setup(rank, world_size):
    # Initialize process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_distributed(rank, world_size, epochs=3):
    # Setup distributed process
    setup(rank, world_size)
    
    # Create model and wrap with DDP
    model = SimpleModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create dataset and dataloader
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Training loop (simplified, without ZeRO for basic functionality)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            inputs = torch.randint(0, 1000, (16, 10)).to(rank)  # Dummy tokenized input
            labels = torch.tensor(batch[1]).to(rank)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if rank == 0:  # Only print on rank 0
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")
    
    cleanup()

if __name__ == "__main__":
    world_size = 2  # Simulate 2 processes (can be run on single machine for demo)
    torch.multiprocessing.spawn(
        train_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
