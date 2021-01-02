import torch
from torch.nn import Module
import torch.nn.functional as F


class DemoModule(Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(5, 5), padding=(2, 2))
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout()
        self.linear1 = torch.nn.Linear(16 * 28 * 28, 100)
        self.linear2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return self.linear2(x)

class MNISTModule(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.maxpool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.maxpool2 = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x).view(-1, 1024)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return self.linear2(x)

#  traced_script_module = torch.jit.script(DemoModule())
traced_script_module = torch.jit.script(MNISTModule())
traced_script_module.save("model.pt")

