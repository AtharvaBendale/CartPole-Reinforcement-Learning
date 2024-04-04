import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_features=5, out_features=25, dtype=torch.float64)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(in_features=25, out_features=16, dtype=torch.float64)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(in_features=16, out_features=10, dtype=torch.float64)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(in_features=10, out_features=5, dtype=torch.float64)
        self.relu4 = nn.ReLU()
        self.lin5 = nn.Linear(in_features=5, out_features=1, dtype=torch.float64)
        self.optimizeer = optim.Adam(params=self.parameters(), lr = 0.001)
        self.criterion = nn.HuberLoss(delta=1)
    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(torch.float64)
        x = x.view(-1, 5)
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        x = self.lin3(x)
        x = self.relu3(x)
        x = self.lin4(x)
        x = self.relu4(x)
        x = self.lin5(x)
        return x
    def _train__instance__(self, train_dataset) -> None:
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        num_epochs = 10
        self.train()
        for epoch in range(num_epochs):
            for batch_id, (data, target) in enumerate(train_loader):
                self.optimizeer.zero_grad()
                output = self(data)
                # print(output.shape, target.shape, type(output), type(target))
                loss = self.criterion(output, target.view(-1,1))
                loss.backward()
                self.optimizeer.step()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    def _save__model__(self, msg : str):
        torch.save(self, f"epoch_{msg}_model.pth")
    def _return__layers__(self):
        return (self.lin1, self.lin2, self.lin3, self.lin4, self.lin5)
    

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label
