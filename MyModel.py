import torch.nn as nn
from BackendClassifier import BackendClassifier

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.backendclassifier = BackendClassifier(input_dim=128, output_dim=1)

    def forward(self, x):
        out = self.backendclassifier(x)
        return out

if __name__ == "__main__":
    pass