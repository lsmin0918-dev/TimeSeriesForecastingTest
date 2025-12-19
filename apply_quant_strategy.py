import json
import re

nb_path = "c:\\AI\\FinalProject\\TimeSeriesForecastingTest\\assignment_notebook.ipynb"

# 1. New Code for MyTradingModel
# Bi-directional GRU, LayerNorm, LeakyReLU, Dropout(0.3), Sigmoid Output
model_code = [
    "# MyTradingModel: 1-Tier Quant Class Implementation\n",
    "import torch.nn as nn\n",
    "import torch\n",
    "\n",
    "class MyTradingModel(nn.Module):\n",
    "    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):\n",
    "        super(MyTradingModel, self).__init__()\n",
    "        self.hidden_size = hidden_size\n",
    "        self.num_layers = num_layers\n",
    "        \n",
    "        # 1. Bi-directional GRU as requested\n",
    "        self.gru = nn.GRU(\n",
    "            input_size, \n",
    "            hidden_size, \n",
    "            num_layers, \n",
    "            batch_first=True, \n",
    "            dropout=dropout, \n",
    "            bidirectional=True\n",
    "        )\n",
    "        \n",
    "        # 2. Stabilization Layers: LayerNorm + LeakyReLU\n",
    "        # Two directions * hidden_size\n",
    "        self.layer_norm = nn.LayerNorm(hidden_size * 2) \n",
    "        self.activation = nn.LeakyReLU(0.01)\n",
    "        self.dropout_layer = nn.Dropout(dropout)\n",
    "        \n",
    "        # 3. Final Output Layer\n",
    "        self.fc = nn.Linear(hidden_size * 2, 1) \n",
    "        self.sigmoid = nn.Sigmoid()\n",
    "        \n",
    "    def forward(self, x):\n",
    "        # Initialize hidden state\n",
    "        # Shape: (num_layers * 2, batch_size, hidden_size)\n",
    "        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)\n",
    "        \n",
    "        # Forward propagate GRU\n",
    "        out, _ = self.gru(x, h0)\n",
    "        \n",
    "        # Decode the hidden state of the last time step\n",
    "        out = out[:, -1, :]\n",
    "        \n",
    "        # Apply stabilization\n",
    "        out = self.layer_norm(out)\n",
    "        out = self.activation(out)\n",
    "        out = self.dropout_layer(out)\n",
    "        \n",
    "        # Final prediction (0~1 probability)\n",
    "        out = self.fc(out)\n",
    "        return self.sigmoid(out)\n"
]

# 2. New Code for simulate_probability_trading
# Position Scaling, 0.5 Threshold, Aggressive
simulation_code = [
    "# Aggressive Trading Logic with Position Scaling\n",
    "def simulate_probability_trading(predictions_prob, actual_prices, dates, \n",
    "                               initial_balance=10000, \n",
    "                               buy_threshold=0.5, \n",
    "                               sell_threshold=0.5, \n",
    "                               transaction_fee=0.001):\n",
    "    \n",
    "    balance = initial_balance\n",
    "    coin_holdings = 0\n",
    "    portfolio_value = [initial_balance]\n",
    "    trades = []\n",
    "    \n",
    "    print(f\"\\n--- Starting Trading Simulation (Aggressive / Position Scaling) ---\")\n",
    "    print(f\"Initial Balance: ${initial_balance:.2f} | Buy Threshold: {buy_threshold}\")\n",
    "    \n",
    "    for i in range(len(predictions_prob)):\n",
    "        current_price = actual_prices[i]\n",
    "        date = dates[i]\n",
    "        prob = predictions_prob[i]\n",
    "        \n",
    "        # --- Position Scaling Logic ---\n",
    "        # Target Ratio dictates how much of the portfolio should be in Bitcoin\n",
    "        # If Probability < 0.5: Target Ratio = 0% (Cash is King)\n",
    "        # If Probability >= 0.5: Target Ratio = Probability (e.g. 0.6 -> 60%, 0.9 -> 90%)\n",
    "        \n",
    "        if prob < 0.5:\n",
    "            target_ratio = 0.0\n",
    "        else:\n",
    "            target_ratio = prob # Direct mapping\n",
    "             \n",
    "        # Calculate current total wealth\n",
    "        current_total_value = balance + (coin_holdings * current_price)\n",
    "        \n",
    "        # Determine amount to Rebalance\n",
    "        target_coin_value = current_total_value * target_ratio\n",
    "        current_coin_value = coin_holdings * current_price\n",
    "        diff_value = target_coin_value - current_coin_value\n",
    "        \n",
    "        # Execute Trade\n",
    "        if diff_value > 0: # Need to BUY\n",
    "            cost = diff_value * (1 + transaction_fee)\n",
    "            if balance >= cost and diff_value > 10: # Minimum trade size $10\n",
    "                amount_to_buy = diff_value / current_price\n",
    "                coin_holdings += amount_to_buy\n",
    "                balance -= cost\n",
    "                trades.append({\"date\": date, \"type\": \"buy\", \"price\": current_price, \"amount\": amount_to_buy, \"cost\": cost})\n",
    "                \n",
    "        elif diff_value < 0: # Need to SELL\n",
    "            sell_value = abs(diff_value)\n",
    "            if sell_value > 10: # Minimum trade size $10\n",
    "                amount_to_sell = sell_value / current_price\n",
    "                # Verify we have enough coin (floating point errors check)\n",
    "                if amount_to_sell > coin_holdings:\n",
    "                    amount_to_sell = coin_holdings\n",
    "                \n",
    "                revenue = amount_to_sell * current_price * (1 - transaction_fee)\n",
    "                coin_holdings -= amount_to_sell\n",
    "                balance += revenue\n",
    "                trades.append({\"date\": date, \"type\": \"sell\", \"price\": current_price, \"amount\": amount_to_sell, \"revenue\": revenue})\n",
    "        \n",
    "        # Track Portfolio Value\n",
    "        portfolio_value.append(balance + (coin_holdings * current_price))\n",
    "\n",
    "    final_value = portfolio_value[-1]\n",
    "    return_rate = (final_value - initial_balance) / initial_balance * 100\n",
    "    \n",
    "    print(f\"Final Balance: ${final_value:.2f} ({return_rate:.2f}%)\")\n",
    "    print(f\"Total Trades: {len(trades)}\")\n",
    "    \n",
    "    return {\n",
    "        \"final_balance\": final_value,\n",
    "        \"return_rate\": return_rate,\n",
    "        \"trades\": trades,\n",
    "        \"portfolio_value\": portfolio_value\n",
    "    }\n"
]

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

mod_counts = 0

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    
    # 3. Apply MyTradingModel Code
    if "class MyTradingModel" in source:
        print("Found MyTradingModel cell. Updating...")
        cell["source"] = model_code
        mod_counts += 1
        
    # 4. Apply Simulation Code
    if "def simulate_probability_trading" in source:
        print("Found simulate_probability_trading cell. Updating...")
        cell["source"] = simulation_code
        mod_counts += 1

if mod_counts > 0:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully updated {mod_counts} cells.")
else:
    print("Error: Could not find target cells to update.")
