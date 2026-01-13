# scopone_a2c ** In Development **

Reinforcement learning agent for the Italian card game **Scopone** (Val Seriana variant), using **PyTorch + Gymnasium** and an **A2C + LSTM** architecture.

This repo currently implements the **2-player** variant with:

- Full game engine and scoring
- Gymnasium environment (`ScoponeEnv`)
- A2C+LSTM policy
- Simple training loop (`train_a2c.py`)

## Installation

```bash
git clone https://github.com/<your-username>/scopone_a2c.git
cd scopone_a2c
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run training

```bash
python train_a2c.py --episodes 20000
```

You should see logs like:

```text
Episode 100 | avg_return=-0.100 | last_len=72 | loss=0.843
Episode 200 | avg_return=0.020 | last_len=86 | loss=0.791
...
```

The return is (team0_score - team1_score) from the perspective of the agent controlling whichever player is on turn.

## Environment

The Gymnasium env is exposed as:

```python
from scopone import ScoponeEnv

env = ScoponeEnv()
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

Observations are flat float32 vectors; actions are integers indexing the card in the current player's hand (0..8).

The environment uses a **self-play** perspective: the same policy controls both players; reward is from the viewpoint of the player who just moved.

## Next steps

- Add 4-player (2 vs 2) variant.
- Allow the policy to explicitly choose sum-combination subsets (not just which card to play).
- Add evaluation against hand-coded heuristic bots.
