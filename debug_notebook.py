
# debug_notebook.py
# This script blindly copies the logic from assignment_notebook.ipynb to check for runtime errors.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import from utils
from utils import (
    load_bitcoin_data,
    create_features,
    prepare_data,
    evaluate_model,
    plot_confusion_matrix,
    device
)

# Settings
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("Starting Debug Process...")

# 1. Load Data
print("Loading Data...")
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

btc_data = load_bitcoin_data(start_date=start_date, end_date=end_date)
btc_features = create_features(btc_data, lookback_days=10)

print(f"Data Loaded: {btc_features.shape}")
print(f"Columns: {btc_features.columns.tolist()}")

# 2. Prepare Data
print("Preparing Data...")
X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
    btc_features, test_size=0.2, validation_size=0.1
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 3. Create Sequences
print("Creating Sequences...")
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

# 4. DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test_seq))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. MyTradingModel (Copied from notebook)
print("Initializing MyTradingModel...")
class MyTradingModel(nn.Module):
    """
    RobustGRU: 2-layer GRU with Dropout & BatchNorm
    """
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super(MyTradingModel, self).__init__()
        
        # 첫 번째 GRU 레이어
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=1, bidirectional=False)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # 두 번째 GRU 레이어 (크기 축소)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        
        # 완전 연결층 (FC)
        self.fc1 = nn.Linear(hidden_size//2, 32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq, feature)
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        
        # BatchNorm을 위한 차원 변경 (batch, hidden, seq)
        out = out.permute(0, 2, 1)
        out = self.bn1(out)
        out = out.permute(0, 2, 1)
        
        out, _ = self.gru2(out)
        
        # 마지막 타임스텝만 사용
        out = out[:, -1, :] 
        
        out = self.dropout2(out)
        out = self.bn2(out)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

my_model = MyTradingModel(
    input_size=X_train_seq.shape[2],
    hidden_size=128,
    dropout=0.3
).to(device)

# 6. Prediction
print("Running Predictions...")
def predict_with_prob(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)
            out = model(batch_X)
            probs.append(out.cpu().numpy())
    return np.vstack(probs).flatten()

my_prob = predict_with_prob(my_model, test_loader)
print(f"Predictions shape: {my_prob.shape}")

# 7. Simulation Prep (The Suspect Part)
print("Preparing Simulation Data...")
try:
    test_start_idx = len(btc_features) - len(y_test) + sequence_length
    print(f"test_start_idx: {test_start_idx}")
    print(f"Sample y_test length: {len(y_test)}")
    print(f"btc_features length: {len(btc_features)}")

    test_prices = btc_features['Close'].iloc[test_start_idx:test_start_idx+len(my_prob)].values
    test_dates = btc_features.index[test_start_idx:test_start_idx+len(my_prob)]
    
    # Check for keys
    if 'RSI_14' not in btc_features.columns:
        raise KeyError("RSI_14 not found in btc_features")
    if 'MACD' not in btc_features.columns:
        raise KeyError("MACD not found in btc_features")
    
    test_rsi = btc_features['RSI_14'].iloc[test_start_idx:test_start_idx+len(my_prob)].values
    test_macd = btc_features['MACD'].iloc[test_start_idx:test_start_idx+len(my_prob)].values
    test_macd_sig = btc_features['MACD_Signal'].iloc[test_start_idx:test_start_idx+len(my_prob)].values
    test_volatility = btc_features['Volatility_10'].iloc[test_start_idx:test_start_idx+len(my_prob)].values
    
    volatility_threshold = np.percentile(test_volatility, 90)

    print("Data Prepared Successfully.")
    
except Exception as e:
    print(f"Error during Data Prep: {e}")
    raise e

# 8. Run Simulation Function
print("Running Simulation...")

def simulate_custom_strategy(probs, prices, dates, rsi, macd, macd_sig, volatilities, vol_thresh, initial_capital=10000):
    cash = initial_capital
    btc = 0
    tx_fee = 0.001
    history = []
    trade_log = []
    
    # 파라미터 : 거래 활성화를 위해 임계값 대폭 낮춤
    BUY_THRESH = 0.50    # 50% 이상이면 매수 검토
    SELL_THRESH = 0.40   # 40% 미만이면 매도 검토
    
    for i in range(len(probs)):
        prob = probs[i]
        price = prices[i]
        vol = volatilities[i]
        r_val = rsi[i]
        m_val = macd[i]
        m_sig = macd_sig[i]
        
        portfolio_val = cash + btc * price
        
        # 마지막 날 전량 매도
        if i == len(probs) - 1:
            if btc > 0:
                cash += btc * price * (1 - tx_fee)
                btc = 0
            history.append(cash)
            continue
            
        # 1. 기본 타겟 비중 계산 (확률 기반)
        # (Prob - 0.4) * 2 => 0.4일때 0, 0.9일때 1.0
        raw_ratio = (prob - 0.4) * 2.0
        target_ratio = min(max(raw_ratio, 0.0), 1.0)
        
        # 2. 보조 지표 보정 (RSI & MACD)
        # RSI < 30 (과매도) -> 강제 매수 신호 (비중 +0.3)
        if r_val < 30:
            target_ratio = max(target_ratio, 0.3)
            target_ratio += 0.2
            
        # RSI > 70 (과매수) -> 비중 축소
        if r_val > 70:
            target_ratio *= 0.5
            
        # MACD 골든크로스 상태 (MACD > Signal) -> 비중 확대
        if m_val > m_sig:
            target_ratio *= 1.2
            # 최소 10%는 가져가도록
            if target_ratio < 0.1: target_ratio = 0.1
        else:
            # 데드크로스 -> 비중 축소
            target_ratio *= 0.8
            
        # 3. 리스크 관리 (초고변동성만 회피)
        if vol > vol_thresh:
            target_ratio *= 0.5
            
        # 최종 비중 클리핑 0~1
        target_ratio = min(max(target_ratio, 0.0), 1.0)
            
        # 4. 리밸런싱
        target_btc_val = portfolio_val * target_ratio
        current_btc_val = btc * price
        diff = target_btc_val - current_btc_val
        
        # 거래 실행
        if diff > 0: # 매수
            # 10달러 이상일 때만 거래
            if diff > 10:
                amount_to_buy_usd = diff
                if amount_to_buy_usd > cash: amount_to_buy_usd = cash
                if amount_to_buy_usd > 0:
                    btc_bought = (amount_to_buy_usd * (1 - tx_fee)) / price
                    btc += btc_bought
                    cash -= amount_to_buy_usd
                    trade_log.append({'date': dates[i], 'action': 'BUY', 'price': price, 'value': amount_to_buy_usd})
        elif diff < 0: # 매도
            if -diff > 10:
                amount_to_sell_usd = -diff
                amount_to_sell_btc = amount_to_sell_usd / price
                if amount_to_sell_btc > btc: amount_to_sell_btc = btc
                if amount_to_sell_btc > 0:
                    cash_gained = amount_to_sell_btc * price * (1 - tx_fee)
                    cash += cash_gained
                    btc -= amount_to_sell_btc
                    trade_log.append({'date': dates[i], 'action': 'SELL', 'price': price, 'value': cash_gained})
            
        history.append(cash + btc * price)

    total_return = (history[-1] - initial_capital) / initial_capital * 100
    
    return {
        'initial_capital': initial_capital,
        'final_value': history[-1],
        'total_return': total_return,
        'portfolio_values': history,
        'num_trades': len(trade_log),
        'total_fees_paid': 0 # Simplified
    }

my_result = simulate_custom_strategy(
    probs=my_prob,
    prices=test_prices,
    dates=test_dates,
    rsi=test_rsi,
    macd=test_macd,
    macd_sig=test_macd_sig,
    volatilities=test_volatility,
    vol_thresh=volatility_threshold
)

print("\n" + "="*70)
print("🚀 Simulation Success!")
print(f"Final Value: ${my_result['final_value']:,.2f}")
print("="*70)
