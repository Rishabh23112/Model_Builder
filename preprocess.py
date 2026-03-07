import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def preprocess_data(df):


    # Features
    x=df.iloc[:,:-1].values

    # Target
    y=df.iloc[:,-1].values


    x=torch.tensor(x, dtype=torch.float32)
    y=torch.tensor(y, dtype=torch.long)

    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)

    train_dataset=TensorDataset(x_train, y_train)

    # DataLoader
    train_loader=DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    return train_loader, x_test, y_test