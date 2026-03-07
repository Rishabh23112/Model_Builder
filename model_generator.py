import torch.nn as nn

class AutoModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(AutoModel, self).__init__()
        self.network=nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.network(x)
    
def build_model(input_size, output_size):
    model=AutoModel(input_size, output_size)
    return model