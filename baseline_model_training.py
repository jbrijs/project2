from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


df = pd.read_csv('./data/AAPL_split_daily_data.csv')
timeseries = df[["close"]].values.astype('float32')
timeseries = timeseries.iloc[::-1]

train_size = int(len(timeseries) * 0.8)
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


class BaseLineMdel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    
model = BaseLineMdel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

# Variables to track the best model
best_test_rmse = float('inf')  # Initialize with a large value

epochs = 2000
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    if epoch % 100 != 0:
        print(f"Epoch: {epoch + 1}")
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    
    # Print RMSE every 100 epochs
    print(f"Epoch: {epoch + 1}, train RMSE: {train_rmse}, test RMSE: {test_rmse}")

    # Save the best model
    if test_rmse < best_test_rmse:
        best_test_rmse = test_rmse
        torch.save(model.state_dict(), 'best_baseline_model.pth')  # Save the model state


best_model = BaseLineMdel()
best_model.load_state_dict(torch.load('best_baseline_model.pth'))
best_model.eval()

# Make predictions using the best model
with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = best_model(X_train)
    y_pred = y_pred[:, -1, :]
    train_plot[lookback:train_size] = best_model(X_train)[:, -1, :]
    
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    test_plot[train_size+lookback:len(timeseries)] = best_model(X_test)[:, -1, :]

# Plot the results
plt.plot(timeseries)
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()
