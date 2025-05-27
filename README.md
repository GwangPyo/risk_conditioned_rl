# Risk-Conditioned Reinforcement Learning A Generalized Approach for Adapting to Varying Risk Measures  
**AAAI 2024**

📄 Paper: [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/29589)

## Overview
This repository implements the methods presented in our AAAI 2024 paper:  
**"Risk-Conditioned Reinforcement Learning: A Generalized Approach for Adapting to Varying Risk Measures."**

The core idea is to condition the policy on a generalized risk parameter,  
enabling a single agent to adapt its behavior across any weighted value-at-risk measures—e.g., CVaR, Wang, CPW, and Power.

## How to Use

### 1. Train the Risk Proposal Network
This step trains the embedding network of risk measures.

```bash
python3 train_risk_proposal.py --save_path=$YOUR_SAVE_PATH 
```

### 2. Train the Agent (SB3-like API)

```python
import gymnasium
from risk_sensitive_rl import GRIPS

env = gymnasium.make("LunarLanderContinuous-v3")
risk_proposal_path = "YOUR_PATH_TO_PRETRAINED_RISK_PROPOSAL_NETWORK"

model = GRIPS(
    env=env,
    policy_type='MlpPolicy',  # Also supports 'CnnPolicy' and 'MultiInputPolicy'
    risk_proposal_path=risk_proposal_path,
    buffer_size=int(1e6),
    lr=3e-4,
    gamma=0.99,
    batch_size=256,
)

model.learn(int(5e6), log_interval=1)
model.save("RiskConditionLunarLander")
```

### 3. SB3 VecEnv Support

```python
import gymnasium
from risk_sensitive_rl import GRIPS
from stable_baselines3.common.vec_env import SubprocVecEnv

env = SubprocVecEnv([
    lambda: gymnasium.make("LunarLanderContinuous-v3")
    for _ in range(4)
])

risk_proposal_path = "YOUR_PATH_TO_PRETRAINED_RISK_PROPOSAL_NETWORK"

model = GRIPS(
    env=env,
    policy_type='MlpPolicy',
    risk_proposal_path=risk_proposal_path,
    buffer_size=int(1e6),
    lr=3e-4,
    gamma=0.99,
    batch_size=256,
)

model.learn(int(5e6), log_interval=1)
model.save("RiskConditionLunarLander")
```

## ⚠️ Action Normalization Warning

The algorithm automatically wraps the Gymnasium environment with an action normalization wrapper.  
However, this is **not supported** for `VecEnv`. You must wrap it manually, like this:

```python
import gymnasium
from risk_sensitive_rl import GRIPS
from stable_baselines3.common.vec_env import SubprocVecEnv
from normalize_action_wrapper import NormalizeActionWrapper

env = SubprocVecEnv([
    lambda: NormalizeActionWrapper(gymnasium.make("LunarLanderContinuous-v3"))
    for _ in range(4)
])

risk_proposal_path = "YOUR_PATH_TO_PRETRAINED_RISK_PROPOSAL_NETWORK"

model = GRIPS(
    env=env,
    ...
)
```
