import torch.nn as nn

class AutoModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(AutoModel, self).__init__()
        hidden = max(32, input_size * 2)
        self.network=nn.Sequential(
            nn.Linear(input_size,hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden,hidden//2),
            nn.ReLU(),


            nn.Linear(hidden//2,output_size)
        )

    def forward(self, x):
        return self.network(x)
    
def build_model(input_size, output_size):
    model=AutoModel(input_size, output_size)
    return model