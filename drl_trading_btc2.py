# full_script_corrected.py
# Import necessary libraries
import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import yfinance as yf
from collections import deque
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler  # For data normalization

# --- Custom Gym Environment for Crypto Trading (Corrected) ---
class CryptoTradingEnv(gym.Env):
    def __init__(self, symbol="BTC-USD", lookback=50):
        super(CryptoTradingEnv, self).__init__()
        self.symbol = symbol
        self.lookback = int(lookback)
        self.initial_balance = float(100000)  # USD

        # Risk Management & Cost Parameters
        self.stop_loss_pct = 0.02     # 2% stop-loss
        self.take_profit_pct = 0.04   # 4% take-profit
        self.transaction_fee = 0.001  # 0.1% per transaction

        # Fetch and process data
        self.data = self._fetch_crypto_data()
        self.scaler = MinMaxScaler()
        self.scaled_data = pd.DataFrame(
            self.scaler.fit_transform(self.data),
            columns=self.data.columns,
            index=self.data.index
        )
        self.original_close_prices = self.data['CLOSE'].copy()

        # Action & Observation Space
        self.action_space = gym.spaces.Discrete(3)  # 0: Buy, 1: Sell, 2: Hold
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.lookback, len(self.data.columns)),
            dtype=np.float32
        )

        # State
        self.balance = 0.0
        self.coins_held = 0.0
        self.current_step = 0
        self.portfolio_value = 0.0

        # Position tracking for correct risk logic
        self.cost_basis = 0.0          # total $ spent on current position (includes buy fees)
        self.avg_entry_price = None    # average entry price for current position

    def _fetch_crypto_data(self):
        # Fetch 5-minute data for the last 60 days
        print("Fetching 5-minute data for the last 60 days...")
        end_date = datetime.today()
        start_date = end_date - timedelta(days=60)

        data = yf.download(self.symbol, start=start_date, end=end_date, interval="5m", auto_adjust=True)
        if data.empty or len(data) < self.lookback + 1:
            raise ValueError(f"Insufficient data fetched for {self.symbol}. Need at least {self.lookback + 1} rows.")

        data = (
            data[['Open', 'High', 'Low', 'Close', 'Volume']]
            .rename(columns={'Open': 'OPEN', 'High': 'HIGH', 'Low': 'LOW', 'Close': 'CLOSE', 'Volume': 'TOTTRDQTY'})
            .astype(float)
            .dropna()
        )
        print("Data fetching complete.")
        return data

    def reset(self):
        self.balance = float(self.initial_balance)
        self.coins_held = 0.0
        self.current_step = int(self.lookback)
        self.portfolio_value = float(self.initial_balance)

        # Reset position metrics
        self.cost_basis = 0.0
        self.avg_entry_price = None

        obs = self._get_observation()
        return obs

    def _get_observation(self):
        # Return lookback window as float32
        window = self.scaled_data.iloc[self.current_step - self.lookback:self.current_step].values
        return window.astype(np.float32, copy=False)

    def _get_current_price(self):
        # Guarantee scalar float (prevents ambiguous Series truth errors)
        price = self.original_close_prices.iloc[int(self.current_step)]
        price = float(price)
        if not np.isfinite(price) or price <= 0.0:
            # Return None to signal unusable price
            return None
        return price

    def _update_portfolio_value(self, current_price: float):
        self.portfolio_value = float(self.balance + self.coins_held * current_price)

    def _maybe_force_exit(self, current_price: float) -> bool:
        """
        Check unrealized P/L against stop-loss/take-profit using avg_entry_price.
        Returns True if we should force-sell.
        """
        if self.coins_held <= 0.0 or self.avg_entry_price is None:
            return False

        if self.avg_entry_price <= 0.0:
            return False

        upnl_pct = (current_price - self.avg_entry_price) / self.avg_entry_price

        if upnl_pct <= -float(self.stop_loss_pct):
            return True
        if upnl_pct >= float(self.take_profit_pct):
            return True
        return False

    def step(self, action):
        # Ensure action is an int
        action = int(action)

        # Get scalar current price; if invalid, advance with hold
        current_price = self._get_current_price()
        if current_price is None:
            # No trading possible; reward is 0 since value unchanged
            prev_portfolio_value = float(self.portfolio_value)
            self.current_step += 1
            done = self.current_step >= (len(self.data) - 1)
            return self._get_observation(), 0.0, done, {
                'total_reward': float(self.portfolio_value - self.initial_balance),
                'total_percent_gain': float((self.portfolio_value - self.initial_balance) / self.initial_balance * 100.0)
            }

        prev_portfolio_value = float(self.portfolio_value)

        # Risk management: override action if stop-loss/take-profit hit
        if self._maybe_force_exit(current_price):
            action = 1  # Force Sell

        # Execute action
        if action == 0:  # Buy
            # Use up to 90% of current balance
            spend = float(self.balance * 0.90)
            if spend > 0.0:
                # Apply fee on notional spend
                fee_buy = float(spend * self.transaction_fee)
                net_spend = float(spend - fee_buy)
                # Coins acquired with net cash after fee
                coins_to_buy = float(net_spend / current_price) if current_price > 0.0 else 0.0

                # Guard against tiny or numerically unstable buys
                if coins_to_buy > 0.0 and np.isfinite(coins_to_buy):
                    # Update wallet: cash down by full spend (fee included), coins up by amount bought
                    self.balance = float(self.balance - spend)
                    self.coins_held = float(self.coins_held + coins_to_buy)

                    # Update cost basis & avg entry (include fee by using full spend)
                    self.cost_basis = float(self.cost_basis + spend)
                    if self.coins_held > 0.0:
                        self.avg_entry_price = float(self.cost_basis / self.coins_held)
                # else: treat as no-op if price invalid or amount too small

        elif action == 1:  # Sell (close entire position for simplicity)
            if self.coins_held > 0.0:
                gross = float(self.coins_held * current_price)
                fee_sell = float(gross * self.transaction_fee)
                proceeds = float(gross - fee_sell)

                self.balance = float(self.balance + proceeds)

                # Reset position
                self.coins_held = 0.0
                self.cost_basis = 0.0
                self.avg_entry_price = None

        else:
            # Hold: no change
            pass

        # Update portfolio value and compute rewards
        self._update_portfolio_value(current_price)
        step_reward = float(self.portfolio_value - prev_portfolio_value)
        total_reward = float(self.portfolio_value - self.initial_balance)
        total_percent_gain = float((total_reward / self.initial_balance) * 100.0)

        # Advance time
        self.current_step += 1
        done = self.current_step >= (len(self.data) - 1)

        info = {
            'total_reward': total_reward,
            'total_percent_gain': total_percent_gain,
            'price': current_price,
            'coins_held': float(self.coins_held),
            'balance': float(self.balance),
            'avg_entry_price': None if self.avg_entry_price is None else float(self.avg_entry_price)
        }

        return self._get_observation(), step_reward, done, info

# --- NEW: Enhanced LSTM-based Neural Network ---
class DQN_LSTM(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN_LSTM, self).__init__()
        # input_shape = (lookback, num_features)
        num_features = input_shape[1]
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        x = F.relu(self.fc1(last_hidden_state))
        return self.fc2(x)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, env, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.n_actions = env.action_space.n

        self.model = DQN_LSTM(env.observation_space.shape, self.n_actions)
        self.target_model = DQN_LSTM(env.observation_space.shape, self.n_actions)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        # state shape (lookback, features) -> convert to batch
        state = torch.FloatTensor(state).unsqueeze(0).contiguous()
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states)).contiguous()
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states)).contiguous()
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# --- Training Loop ---
def train_dqn(episodes=10, batch_size=32):
    env = CryptoTradingEnv(symbol="BTC-USD")
    agent = DQNAgent(env)

    for episode in range(episodes):
        state = env.reset()
        done = False
        info = {}

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn(batch_size)
            state = next_state

        agent.update_target_model()

        print(f"Episode {episode + 1}/{episodes} Summary | "
              f"Final P/L: ${info.get('total_reward', 0):.2f} ({info.get('total_percent_gain', 0):.4f}%) | "
              f"Epsilon: {agent.epsilon:.3f}")

if __name__ == "__main__":
    train_dqn(episodes=10)
