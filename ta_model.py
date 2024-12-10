import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load dataset
df = pd.read_csv('./data/AAPL_preprocessed_data.py')

# Drop 'time_stamp' column and reverse order
timeseries = df.drop(columns=['time_stamp']).iloc[::-1]

# Train-test split
train_size = int(len(timeseries) * 0.9)
train, test = timeseries[:train_size], timeseries[train_size:]

#Choose scalar function
def choose_scaler(columns, mm_features, ss_features):
    for col in columns:
        mean = train[col].mean()
        std = train[col].std()
        min_val, max_val = train[col].min(), train[col].max()

        price_columns = ['open', 'high', 'low', 'close']
        if col in price_columns:
            ss_features.append(col)
        elif min_val >= 0 and max_val <= 100: # Likely a bounded indicator, use minmax
            mm_features.append(col)
        elif abs(mean) < std: # Likely unbounded, use StandardScaler
            ss_features.append(col)
        else:
            mm_features.append(col)

        return mm_features, ss_features
    
mm_features, ss_features = choose_scaler(train.columns, [], [])

# Initialize scalers
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Scale features for training and testing
train_scaled = train.copy()
test_scaled = test.copy()

# Apply MinMaxScaler to selected features
for col in mm_features:
    train_scaled[col] = min_max_scaler.fit_transform(train[[col]])
    test_scaled[col] = min_max_scaler.transform(test[[col]])

# Apply StandardScaler to selected features
for col in ss_features:
    train_scaled[col] = standard_scaler.fit_transform(train[[col]])
    test_scaled[col] = standard_scaler.transform(test[[col]])


train = train_scaled.values.astype('float32')
test = test_scaled.values.astype('float32')

# Create dataset
def create_dataset(dataset, lookback, close_col):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i: i + lookback]
        target = dataset[i + lookback, close_col]
        X.append(feature)
        y.append(target)
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))


lookback = 10
close_col = df.columns.get_loc('close')
X_train, y_train = create_dataset(train, lookback, close_col)
X_test, y_test = create_dataset(test, lookback, close_col)


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, output_size, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_features, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use the last time step
        return out


# Initialize the model
num_features = X_train.shape[2]
hidden_dim = 50
num_layers = 1
output_size = 1

model = LSTMModel(num_features, hidden_dim, num_layers, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# DataLoader
loader = DataLoader(TensorDataset(X_train, y_train),
                    shuffle=False, batch_size=8)

# Train the model
epochs = 1000
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_batch = y_batch.view(-1, 1)  # Reshape target to [batch_size, 1]
        y_pred = model(X_batch)  # Model output is [batch_size, 1]
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train)
        y_train_actual = y_train.view(-1, 1)  # Reshape target to [batch_size, 1]
        train_rmse = np.sqrt(loss_fn(y_train_pred, y_train_actual).item())

        y_test_pred = model(X_test)
        y_test_actual = y_test.view(-1, 1)  # Reshape target to [batch_size, 1]
        test_rmse = np.sqrt(loss_fn(y_test_pred, y_test_actual).item())

    print(f"Epoch: {epoch + 1}/{epochs}, Train Loss: {train_rmse:.4f}, Test Loss: {test_rmse:.4f}")

# Save the model
torch.save(model.state_dict(), "lstm_model.pth")

# Make predictions
with torch.no_grad():
    train_predictions = model(X_train)
    test_predictions = model(X_test)

    # Prepare plots
    train_plot = np.full_like(timeseries[:, 0], np.nan)
    test_plot = np.full_like(timeseries[:, 0], np.nan)

    # Fill predictions in respective ranges
    train_plot[lookback:train_size] = train_predictions.squeeze()
    test_plot[train_size + lookback:] = test_predictions.squeeze()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(timeseries[:, close_col], label="Actual Close Price", color="blue")
plt.plot(train_plot, label="Train Predictions", color="red")
plt.plot(test_plot, label="Test Predictions", color="green")
plt.legend()
plt.title("Actual vs Predicted Close Prices")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.grid()
plt.savefig("lstm_predictions.png")
plt.show()

