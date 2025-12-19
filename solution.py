import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import utils
import sys

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 1. Data Preparation (Same as notebook)
# ==========================================
print("\n[Step 1] Loading and Preparing Data...")
# Load data
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")
btc_data = utils.load_bitcoin_data(start_date=start_date, end_date=end_date)
btc_features = utils.create_features(btc_data, lookback_days=10)

# Split and Scale
X_train, X_val, X_test, y_train, y_val, y_test = utils.prepare_data(
    btc_features, test_size=0.2, validation_size=0.1
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Sequence Creation
sequence_length = 30

def create_sequences(X, y, seq_len=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, sequence_length)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val.values, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, sequence_length)

# DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test_seq))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==========================================
# 2. Model Definition (RobustGRU)
# ==========================================
print("\n[Step 2] Defining RobustGRU Model...")

class MyTradingModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super(MyTradingModel, self).__init__()
        
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=1, bidirectional=False)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Second layer smaller
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        
        self.fc1 = nn.Linear(hidden_size//2, 32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, seq, feature)
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        
        # Batch Norm checks: (batch, feature, seq) for 1d? No, BN1d inputs (N, C) or (N, C, L)
        # We need to permute for BN if we want to normalize features across sequence? 
        # Actually standard practice for RNN is often LayerNorm, but let's stick to BN on the output dim.
        # Shape is (batch, seq, hidden). Permute to (batch, hidden, seq)
        out = out.permute(0, 2, 1)
        out = self.bn1(out)
        out = out.permute(0, 2, 1)
        
        out, _ = self.gru2(out)
        # Take last time step
        out = out[:, -1, :] 
        
        out = self.dropout2(out)
        out = self.bn2(out)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# ==========================================
# 3. Training
# ==========================================
print("\n[Step 3] Training Model...")

# Reusing the training loop from notebook to be precise
def train_model_custom(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # Added weight_decay
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

model = MyTradingModel(input_size=X_train_seq.shape[2]).to(device)
model = train_model_custom(model, train_loader, val_loader, epochs=100, lr=0.0005, patience=20)

# ==========================================
# 4. Prediction
# ==========================================
print("\n[Step 4] Making Predictions...")
def predict_with_prob(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)
            out = model(batch_X)
            probs.append(out.cpu().numpy())
    return np.vstack(probs).flatten()

test_probs = predict_with_prob(model, test_loader)

# Prepare Test Data for Simulation
test_start_idx = len(btc_features) - len(y_test) + sequence_length
test_prices = btc_features["Close"].iloc[test_start_idx:test_start_idx+len(test_probs)].values
test_dates = btc_features.index[test_start_idx:test_start_idx+len(test_probs)]

# Volatility for scaling (need to align this too)
# Volatility_10 is in features
test_volatility = btc_features["Volatility_10"].iloc[test_start_idx:test_start_idx+len(test_probs)].values
volatility_threshold = np.percentile(test_volatility, 80) # Top 20% volatility

# ==========================================
# 5. Custom Strategy Simulation
# ==========================================
print("\n[Step 5] Simulating Strategy...")

def simulate_custom_strategy(probs, prices, dates, volatilities, vol_thresh, initial_capital=10000):
    cash = initial_capital
    btc = 0
    tx_fee = 0.001
    
    history = []
    
    # Thresholds
    BUY_THRESH = 0.65 # Conservative
    SELL_THRESH = 0.35
    
    for i in range(len(probs)):
        prob = probs[i]
        price = prices[i]
        vol = volatilities[i]
        
        portfolio_val = cash + btc * price
        
        # Last day sell all
        if i == len(probs) - 1:
            if btc > 0:
                cash += btc * price * (1 - tx_fee)
                btc = 0
            history.append(cash)
            continue
            
        # Target Ratio Calculation
        if prob > BUY_THRESH:
            # High confidence buy
            # Scale by confidence: 0.65 -> 0.3, 1.0 -> 1.0
            # Normalized: (0.65 - 0.5) * 2 = 0.3... wait.
            # Let's say: (Prob - 0.5) * 2. 
            # If prob 0.7 => 0.4 ratio. If prob 0.9 => 0.8 ratio.
            raw_ratio = (prob - 0.5) * 2.0
            target_ratio = min(max(raw_ratio, 0.0), 1.0)
            
            # Volatility Penalty
            if vol > vol_thresh:
                target_ratio *= 0.5 # Halve position in high volatility
                
        elif prob < SELL_THRESH:
            target_ratio = 0.0
        else:
            # Hold zone - maintain current alignment if possible, or just hold current BTC
            # But "Hold" in these rebalancing strategies usually means "Keep current RATIO"?
            # Or "Don't Trade". 
            # If we don't trade, ratio changes naturally as price moves.
            current_value_ratio = (btc * price) / portfolio_val if portfolio_val > 0 else 0
            target_ratio = current_value_ratio # Don't change anything
            
        # Rebalancing
        # Note: To avoid tiny trades, we can add a buffer, but let's stick to logic
        
        target_btc_val = portfolio_val * target_ratio
        current_btc_val = btc * price
        
        diff = target_btc_val - current_btc_val
        
        # Minimum trade buffer to avoid dusting (e.g. $10)
        if abs(diff) < 50: 
            history.append(portfolio_val)
            continue
            
        if diff > 0: # Buy
            amount_to_buy_usd = diff
            if amount_to_buy_usd > cash: amount_to_buy_usd = cash
            btc_bought = (amount_to_buy_usd * (1 - tx_fee)) / price
            btc += btc_bought
            cash -= amount_to_buy_usd
        elif diff < 0: # Sell
            amount_to_sell_usd = -diff
            amount_to_sell_btc = amount_to_sell_usd / price
            if amount_to_sell_btc > btc: amount_to_sell_btc = btc
            
            cash_gained = amount_to_sell_btc * price * (1 - tx_fee)
            cash += cash_gained
            btc -= amount_to_sell_btc
            
        history.append(cash + btc * price)

    total_return = (history[-1] - initial_capital) / initial_capital * 100
    return total_return, history[-1]

my_return, my_final = simulate_custom_strategy(test_probs, test_prices, test_dates, test_volatility, volatility_threshold)

# Calculate Buy and Hold
bh_btc = (10000 * 0.999) / test_prices[0]
bh_final = bh_btc * test_prices[-1] * 0.999
bh_return = (bh_final - 10000) / 10000 * 100

print("\n" + "="*50)
print(f"FINAL RESULTS")
print("="*50)
print(f"Buy and Hold Return: {bh_return:.2f}% (${bh_final:,.2f})")
print(f"My Strategy Return:  {my_return:.2f}% (${my_final:,.2f})")
print(f"Excess Return:       {my_return - bh_return:.2f}%p")
print("="*50)

if my_return > bh_return:
    print("SUCCESS: Strategy beat the benchmark! 🚀")
else:
    print("FAILURE: Strategy failed to beat benchmark. Needs tuning.")
