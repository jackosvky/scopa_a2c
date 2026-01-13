# ScopaBergamasca_a2c *** IN DEVELOPMENT

Reinforcement learning agents for the Italian card game **ScopaBergamasca** (Val Seriana variant), using **PyTorch + Gymnasium** and an **A2C + LSTM** architecture.

This repository currently implements the **2-player** variant (1 vs 1, each its own “team”) with:

- A full game engine and scoring for the Val Seriana rules.
- A Gymnasium environment (`ScopaBergamascaEnv`) with a **human-like observation**:
  - 9 cards in hand as a suit×rank tensor.
  - Up to 10 cards on the table as a suit×rank tensor.
  - The previous card played, and whether it captured or scored a Scopa.
  - Aggregate scoring-related stats (coins, 7s, captured cards, scopas).
- A simple A2C + LSTM training loop in `train_a2c.py`.

The aim is to let agents learn strategies comparable to human play while only seeing information humans would have access to.

---

## Game rules (summary of current implementation)

We implement ScopaBergamasca with the **Val Seriana** rules, 2-player version:

- **Deck**: 40 cards, 4 suits (coins, cups, swords, clubs), ranks 1–7 + jack(8), queen(9), king(10).
- **Dealing**:
  - Shuffle the deck.
  - Deal 9 cards to each player, 4 cards face up on table.
  - After those 18 cards are played, deal another 9+9 (second “episode”).
- **Turn order**:
  - Two players: Player 0 starts, then Player 1, alternating.

### Capturing rules

On your turn, you:

1. **Play one card from your hand**.
2. The game engine determines the capture:

   - **Ace (1)**:
     - If there are cards on the table, the ace **captures all cards on the table**.
     - If the table is empty, the ace is simply placed on the table.
     - **No Scopa** is awarded for clearing with an ace.

   - **Non-ace**:
     1. **Single-card capture priority**:
        - If there is any card on the table with the same rank as the played card,
          you MUST capture exactly one such card (the engine picks the first).
     2. If no equal-rank card is present:
        - You may capture a **subset of table cards whose ranks sum to the rank of the played card**.
        - In this implementation (Phase 1), when multiple subsets exist, the engine uses a deterministic rule:
          - Maximize the number of **coins** captured.
          - Tie-break by subset size.
          - Tie-break by lexicographic order of card IDs.

   - If no capture is possible (no equal rank, no sum), the played card remains on the table.

3. If a non-ace capture leaves the table **empty**, that is a **Scopa** (+1 point), except if this is the **very last card of the entire game**, in which case no Scopa is counted (engine corrects this at game end).

4. **Last capture**:
   - At the end of the game, any remaining cards on the table go to the **last player who made a capture**.

### Scoring

At the end of a full 40-card game, both players’ piles are evaluated:

Each player/team scores:

1. **Most coins** (`suit=coins`):
   - +1 if you have more coins than the opponent (strict majority).
   - No point if tied (e.g., 5 vs 5 in 10-coin games).

2. **Coin sequence including 1,2,3**:
   - Consider the **longest contiguous rank sequence of coins** you have
     that includes ranks 1, 2, and 3.
   - If its length is `L ≥ 3`, you score **+L points**.
   - Example:
     - Coins [1,2,3] → +3
     - Coins [1,2,3,4,5] → +5
     - Coins [2,3,4,5] (no 1) → +0

3. **Most 7s**:
   - +1 if you have more rank-7 cards than your opponent.
   - No point if tied.

4. **Most total cards**:
   - +1 if you have more captured cards (total) than your opponent.
   - No point if tied (20 vs 20).

5. **7 of coins**:
   - +1 if you have the **7 of coins** in your pile.

6. **Scopa**:
   - +1 per Scopa recorded during the game (except the final-card Scopa, which is removed).

**Final reward** for RL:

- For training, we define:
  - `reward_team0 = (score_team0 - score_team1)`,
  - `reward_team1 = -reward_team0`.
- Each player’s RL reward is their team reward (here, one player per team).

---

## Observation (agent input)

The agent sees **only what a human player would see** on their turn, plus some simple, aggregated stats:

The Gym observation is a `spaces.Dict`:

```python
{
  "hand":           Box(shape=(9, 4, 10), 0..1),
  "hand_mask":      Box(shape=(9,),       0..1),
  "table":          Box(shape=(10, 4, 10),0..1),
  "table_mask":     Box(shape=(10,),      0..1),
  "prev_card":      Box(shape=(4, 10),    0..1),
  "prev_captured":  Box(shape=(1,),       0..1),
  "prev_scopa":     Box(shape=(1,),       0..1),
  "current_player": Box(shape=(2,),       0..1),
  "episode":        Box(shape=(2,),       0..1),
  "remaining_deck": Box(shape=(1,),       0..1),
  "score_stats":    Box(shape=(8,),       0..1),
}
```

### Details

- **`hand`**: `[9, 4, 10]`
  - Up to 9 hand slots.
  - Each slot is a one-hot over (suit, rank):
    - `hand[i, suit, rank-1] = 1.0` if that slot holds this card.
  - Unused slots are all zeros.
- **`hand_mask`**: `[9]`
  - 1 if that hand slot is occupied, 0 if empty.
  - Used to mask invalid actions.

- **`table`**: `[10, 4, 10]`
  - Up to 10 visible cards on the table, tensor slot per card.
- **`table_mask`**: `[10]`
  - 1 if that table slot is occupied.

- **`prev_card`**: `[4, 10]`
  - One-hot of the **last card played** (by either player).
  - All zeros if no previous card (start of game).

- **`prev_captured`**: `[1]`
  - 1.0 if the last played card captured any cards, else 0.0.

- **`prev_scopa`**: `[1]`
  - 1.0 if the last move was recorded as a Scopa (engine later removes last-card Scopa when appropriate).

- **`current_player`**: `[2]`
  - One-hot: [1, 0] if Player 0 is on turn, [0, 1] if Player 1 is on turn.
  - The agent always “is” the current player; the same policy is used for both.

- **`episode`**: `[2]`
  - One-hot: first or second 9-card episode.

- **`remaining_deck`**: `[1]`
  - `len(deck) / 40.0` in [0, 1].

- **`score_stats`**: `[8]` (from the current player’s perspective)
  - `[ own_coins / 10,
      own_sevens / 4,
      own_total / 40,
      own_scopas / 10,
      opp_coins / 10,
      opp_sevens / 4,
      opp_total / 40,
      opp_scopas / 10 ]`

This is a **partial, human-like view**: the agent is not told exactly which cards are in the deck or in the opponent’s hand; it must infer that over time via its LSTM.

For training, we flatten this dict to a single vector (length 834) before feeding it to the A2C+LSTM network.

---

## Action space

The action space is:

- `Discrete(9)`: index of the card to play from the **current hand** (slot 0–8).

The environment:

- Uses `hand_mask` to invalidate actions for empty slots (logits set to `-inf` before softmax).
- Once a card is chosen, it applies the **deterministic capture rules**:
  - Ace rule.
  - Single-card capture priority.
  - Heuristic sum-combination if needed.

Future extensions could let the agent directly choose which sum-combination subset to take, but this first version focuses solely on **which card to play**.

---

## Code structure

Recommended layout:

```text
ScopaBergamasca_a2c/
  ScopaBergamasca/
    __init__.py
    cards.py
    game.py
    env.py
    utils.py        # optional helpers (e.g., make_env)
  train_a2c.py
  requirements.txt
  README.md
```

### `ScopaBergamasca/cards.py`

- Defines `Suit` and `Card` (suit, rank, id).

### `ScopaBergamasca/game.py`

- Implements the full 2-player ScopaBergamasca game:
  - Dealing, turn logic, capturing, Scopa, scoring.
  - Tracks:
    - `last_played_card`
    - `last_move_captured`
    - `last_move_was_scopa`
    - `last_move_player`
  - Returns final team-score differences as RL rewards.

### `ScopaBergamasca/env.py`

- Wraps `ScopaBergamascaGame2P` as a **Gymnasium** environment:
  - `action_space = Discrete(9)`
  - `observation_space = Dict(...)` as above.
- Single-agent **self-play** view:
  - At each step, the agent controls whichever player is on turn.
  - The reward is from the perspective of the player who just acted.

### `train_a2c.py`

- Defines:
  - `A2CLSTMPolicy`: MLP → LSTM → policy+value heads.
  - `flatten_obs(obs_dict)`: turns the dict into a flat vector for the network.
  - `run_episode(...)`: runs a single episode, collects transitions, computes returns/advantages.
  - `train(...)`: basic A2C loop over multiple episodes.

---

## Installation

```bash
git clone https://github.com/<your-username>/ScopaBergamasca_a2c.git
cd ScopaBergamasca_a2c

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

`requirements.txt` should contain:

```text
gymnasium>=0.29.0
numpy>=1.23.0
torch>=2.0.0
```

---

## Running training

```bash
python train_a2c.py --episodes 20000
```

You should see logs like:

```text
Episode 100 | avg_return=-0.120 | last_len=80 | loss=0.842
Episode 200 | avg_return=0.030  | last_len=85 | loss=0.799
...
```

- `avg_return` is the moving average of (team0_score - team1_score) from the perspective of the acting player.
- Over many episodes, you should see learning progress if hyperparameters are reasonable.

---

## Using the environment manually

You can also interact with the environment directly:

```python
from ScopaBergamasca import ScopaBergamascaEnv

env = ScopaBergamascaEnv()
obs, info = env.reset()
done = False
truncated = False

while not (done or truncated):
    # Random policy (for testing)
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
```

This will print out:
- Current player’s hand,
- The table,
- Each player’s captured cards and Scopas.

---

## Next steps and extensions

Some natural extensions to this project:

1. **4-player (2 vs 2) ScopaBergamasca**
   - Add 4-player dealing and seating.
   - Modify the environment to handle 4 players and team-based rewards.
   - Still use a shared policy, now conditioning on player/team id.

2. **Agent chooses sum-combinations**
   - Extend action space to also select the subset of table cards for sum captures.
   - Use either:
     - A factorized action (card choice + subset choice), or
     - Enumerated legal subsets per state.

3. **Heuristic opponent vs learning agent**
   - Implement a simple heuristic bot (prioritizing coins, 7 of coins, avoiding giving Scopa, etc.).
   - Train the agent against the heuristic to study human-comparable play.

4. **Analysis tools**
   - Save rollouts and inspect:
     - How often the agent targets coins vs 7s.
     - How it values Scopa opportunities.
   - Compare to human ScopaBergamasca strategies.

---

If you make substantial changes (e.g., 4-player version or subset-choice actions), consider documenting the new observation/action spaces and training setup here as well.
