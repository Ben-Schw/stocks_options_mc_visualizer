import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

"""
Downloads and returns cleaned price and volume data for a given ticker.

Parameters:
ticker (str) - Stock symbol
start - Start date
end - End date

Return:
pd.DataFrame - DataFrame with columns ["Close", "Volume"]
"""
def load_data(ticker: str, start="2015-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df[["Close", "Volume"]].dropna()
    return df

"""
Builds rolling window sequences for supervised learning.

Parameters:
X (np.ndarray) - Feature matrix (time x features)
y (np.ndarray) - Target array (time x 1) or (time,)
lookback (int) - Number of past time steps per input sequence

Return:
tuple[np.ndarray, np.ndarray] - (X_sequences, y_targets)
"""
def make_sequences_xy(X: np.ndarray, y: np.ndarray, lookback: int):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i]) 
        ys.append(y[i])                
    return np.array(Xs), np.array(ys)

"""
Trains the model for one epoch over the provided DataLoader.

Parameters:
model - PyTorch model
loader - DataLoader providing batches
optimizer - Optimizer instance
loss_fn - Loss function
device - Torch device

Return:
float - Average loss over the epoch
"""
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


"""
Evaluates the model on a DataLoader and returns regression metrics.

Parameters:
model - PyTorch model
loader - DataLoader providing batches
loss_fn - Loss function
device - Torch device

Return:
dict - Metrics dictionary with keys {"mse", "rmse", "mae"}
"""
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    abs_err_sum = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)

        loss = loss_fn(pred, y)
        total_loss += loss.item() * X.size(0)

        abs_err_sum += torch.abs(pred - y).sum().item()

    mse = total_loss / len(loader.dataset)
    mae = abs_err_sum / len(loader.dataset)
    rmse = np.sqrt(mse)
    return {"mse": mse, "rmse": rmse, "mae": mae}

"""
Generates predictions for all batches in a DataLoader and returns stacked arrays.

Parameters:
model - PyTorch model
loader - DataLoader providing batches
device - Torch device

Return:
tuple[np.ndarray, np.ndarray] - (predictions, targets)
"""
@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, ys = [], []
    for X, y in loader:
        X = X.to(device)
        pred = model(X).cpu().numpy()
        preds.append(pred)
        ys.append(y.numpy())
    return np.vstack(preds), np.vstack(ys)

class SequenceDataset(Dataset):
    """
    Wraps numpy arrays as a PyTorch Dataset for sequence regression.

    Parameters:
    X (np.ndarray) - Input sequences
    y (np.ndarray) - Targets

    Return:
    None
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)

    """
    Returns the number of samples in the dataset.

    Parameters:
    None

    Return:
    int - Dataset length
    """
    def __len__(self):
        return self.X.shape[0]

    """
    Returns one (X, y) sample pair.

    Parameters:
    idx - Index of the sample

    Return:
    tuple - (X[idx], y[idx])
    """
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMRegressor(nn.Module):
    """
    Defines an LSTM-based regression model that maps sequences to a single output value.

    Parameters:
    input_size - Number of features per time step
    hidden_size - LSTM hidden size
    num_layers - Number of LSTM layers
    dropout - Dropout probability between LSTM layers

    Return:
    None
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    """
    Runs a forward pass on a batch of sequences and returns predictions.

    Parameters:
    x - Input tensor of shape (batch, time, features)

    Return:
    torch.Tensor - Output tensor of shape (batch, 1)
    """
    def forward(self, x):
        out, _ = self.lstm(x)      
        last = out[:, -1, :]       
        return self.fc(last)   
    

if __name__ == "__main__":
    df = load_data(ticker="AAPL", start="2015-01-01")

    df["log_ret_1d"] = np.log(df["Close"]).diff()
    df["log_ret_5d"] = np.log(df["Close"].shift(-5) / df["Close"])  # Target

    df["vol_20"] = df["log_ret_1d"].rolling(20).std()
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["vol_chg"] = np.log(df["Volume"]).diff()

    df = df.dropna()
    print(df.head())

    features = ["log_ret_1d", "vol_20", "ma_5", "ma_20", "vol_chg"]
    target = "log_ret_5d"  

    X_all = df[features].values             
    y_all = df[target].values.reshape(-1, 1)

    lookback = 60

    split = int(len(df) * 0.8)
    X_train_raw, X_test_raw = X_all[:split], X_all[split:]
    y_train_raw, y_test_raw = y_all[:split], y_all[split:]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train_raw)
    X_test  = x_scaler.transform(X_test_raw)

    y_train = y_scaler.fit_transform(y_train_raw)
    y_test  = y_scaler.transform(y_test_raw)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    X_train_seq, y_train_seq = make_sequences_xy(X_train, y_train, lookback)

    X_test_full = np.vstack([X_train[-lookback:], X_test])
    y_test_full = np.vstack([y_train[-lookback:], y_test])

    X_test_seq, y_test_seq = make_sequences_xy(X_test_full, y_test_full, lookback)

    print("X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)
    print("X_test_seq :", X_test_seq.shape,  "y_test_seq :", y_test_seq.shape)

    batch_size = 64
    train_loader = DataLoader(SequenceDataset(X_train_seq, y_train_seq), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(SequenceDataset(X_test_seq, y_test_seq), batch_size=batch_size, shuffle=False) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = LSTMRegressor(input_size=len(features), hidden_size=64, num_layers=2, dropout=0.1).to(device)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    best_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        metrics = evaluate(model, test_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train MSE: {train_loss:.6f} | "
            f"test MAE: {metrics['mae']:.6f} | "
            f"test RMSE: {metrics['rmse']:.6f}"
        )

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best model. Best scaled MAE:", best_mae)

    pred_scaled, y_scaled = predict_all(model, test_loader, device)

    pred_ret = y_scaler.inverse_transform(pred_scaled)  # log returns
    true_ret = y_scaler.inverse_transform(y_scaled)

    mae_ret = mean_absolute_error(true_ret, pred_ret)
    print("Test MAE (log return):", mae_ret)

    direction_acc = (np.sign(true_ret) == np.sign(pred_ret)).mean()
    print("Directional accuracy:", direction_acc)

    plt.figure(figsize=(12, 5))
    plt.plot(true_ret, label="True")
    plt.plot(pred_ret, label="Predicted")
    plt.xlabel("Test samples")
    plt.title("AAPL LSTM - 5-day log return prediction")
    plt.ylabel("5-day log return")
    plt.legend()
    plt.tight_layout()
    plt.show()
