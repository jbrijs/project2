import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import argparse


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(
            0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(
            0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


'''
BASELINE MODEL

Model trained on just Open, High, Low, Volume, and finding Close
'''


def train_test_split_simple(ticker, train_ratio=0.8):
    df = pd.read_csv(f"./data/{ticker}_split_daily_data.csv")

    # Reverse df so it is in order
    df = df.iloc[::-1].reset_index(drop=True)

    X = df.drop(['time_stamp', 'close'], axis=1)

    y = df['close']

    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).view(-1, 1)

    # Split into training and testing sets
    train_size = int(len(X_tensor) * train_ratio)
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

    return X_train, X_test, y_train, y_test


def save_model(model, ticker):
    torch.save(model.state_dict(), f"./models/{ticker}_model.pth")


def train_simple_model(X_train, X_test, y_train, y_test, ticker, batch_size=32, epochs=1000):

    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(
        input_size=X_train.shape[1], hidden_size=50, num_layers=2, output_size=1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for i, (data, target) in enumerate(train_loader):
            data = data.unsqueeze(2)

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(
                    f'Epoch: {epoch+1}, Batch: {i+1} Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        X_test = X_test.unsqueeze(2)
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)

    print(f'Test Loss: {test_loss.item():.4f}')

    # Convert to NumPy for analysis
    predictions_np = predictions.squeeze().cpu().numpy()
    y_test_np = y_test.squeeze().cpu().numpy()

    # Save to CSV
    results_df = pd.DataFrame({
        'Predictions': predictions_np,
        'Ground Truth': y_test_np
    })
    results_df.to_csv(f"./results/{ticker}_predictions.csv", index=False)
    print(f"Predictions saved to ./results/{ticker}_predictions.csv")

    print(f'Saving model...')
    save_model(model, ticker)
    print('Model saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model for a specific ticker')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    args = parser.parse_args()
    ticker = args.ticker
    X_train, X_test, y_train, y_test = train_test_split_simple(ticker)
    train_simple_model(X_train=X_train, X_test=X_test,
                       y_train=y_train, y_test=y_test, ticker=ticker)
