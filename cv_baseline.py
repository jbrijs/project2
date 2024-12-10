from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit


df = pd.read_csv('./data/AAPL_split_daily_data.csv')
timeseries = df[["close"]].iloc[::-1].values.astype('float32')


# Decrease size of test set, so the model can learn newer data
train_size = int(len(timeseries) * 0.9)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    X, y = [], []
    
    for i in range(len(dataset) - lookback):
        feature = dataset[i: i + lookback]
        target = dataset[i + 1: i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

lookback = 10

X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

tscv = TimeSeriesSplit(n_splits=5)

class BaseLineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    
def train_model(X_train, y_train, model, optimizer, loss_fn, epochs=100, batch_size=8):
    model.train()
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.unsqueeze(-1)
            y_batch = y_batch

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch: {epoch + 1}/{epochs}, Loss: {epoch_loss / len(loader):.6f}")

    torch.save(model.state_dict(), 'cv_baseline_model.pth')

    
# Cross-validation
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    print(f"Fold {fold + 1}")
    train_data, val_data = X_train[train_idx], X_train[val_idx]
    train_labels, val_labels = y_train[train_idx], y_train[val_idx]

    model = BaseLineModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_model(train_data, train_labels, model, optimizer, loss_fn, epochs=100)

    torch.save(model.state_dict(), 'cv_baseline_model.pth')


model.eval()

with torch.no_grad():
    # Reshape inputs for LSTM ([samples, lookback, features])
    X_train_input = X_train.unsqueeze(-1)  # Add feature dimension
    X_test_input = X_test.unsqueeze(-1)

    # Predictions for training data
    train_predictions = model(X_train_input).squeeze(-1).numpy()
    train_plot = np.ones_like(timeseries) * np.nan
    train_plot[lookback:train_size] = train_predictions[:, -1]

    # Predictions for testing data
    test_predictions = model(X_test_input).squeeze(-1).numpy()
    test_plot = np.ones_like(timeseries) * np.nan
    test_plot[train_size + lookback:] = test_predictions[:, -1]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(timeseries, label='Actual Close Price', c='b', linewidth=1)
plt.plot(train_plot, label='Train Predictions', c='r', linestyle='--')
plt.plot(test_plot, label='Test Predictions', c='g', linestyle='--')

plt.title("Time Series Predictions")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.grid()

# Save and show the plot
plt.savefig('baseline_predictions_plot.png')
plt.show()

