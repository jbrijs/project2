import torch 
import torch.nn as nn
import torch.optim as optim
import pandas as pd

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
'''
BASELINE MODEL

Model trained on just Open, High, Low, Volume, and finding Close
'''

def train_test_split_simple(ticker):
    df = pd.read_csv(f"./data/{ticker}_split_daily_data.csv")


    # Reverse df so it is in order
    df = df.iloc[::-1].reset_index(drop=True)

    X = df.drop(['time_stamp', 'close'], axis=1)

    y = df['close']

    X_tensor = torch.tensor(X.to_numpy())
    y_tensor = torch.tensor(y.to_numpy())

    return X_tensor, y_tensor
    

if __name__ == '__main__':
    train_test_split_simple('AAPL')

def train_simple_model(X, y):
    model = LSTMModel(input_size=X.size, hidden_size=50, num_layers=50, output_size=1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)