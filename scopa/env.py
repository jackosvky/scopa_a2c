import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

from .cards import Card, Suit
from .game import ScoponeGame2P


MAX_HAND_SIZE = 9
MAX_TABLE_SIZE = 10
N_SUITS = 4
N_RANKS = 10


class ScoponeEnv(gym.Env):
    """
    Gymnasium env for 2-player Scopone.

    - Single-agent interface with self-play:
      the agent always controls "current_player" (0 or 1),
      and the same policy is used for both players.
    - Observations are human-like:
      - Hand as [9, 4, 10] (up to 9 cards)
      - Table as [10, 4, 10] (up to 10 cards)
      - Previous card, whether it captured / was scopa
      - Score-like stats and progress info
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self._rng = np.random.RandomState(seed)
        self.game = ScoponeGame2P()

        # Action: index of card in hand (0..MAX_HAND_SIZE-1)
        self.action_space = spaces.Discrete(MAX_HAND_SIZE)

        # Structured dict observation space
        self.observation_space = spaces.Dict({
            "hand":          spaces.Box(0.0, 1.0, shape=(MAX_HAND_SIZE, N_SUITS, N_RANKS), dtype=np.float32),
            "hand_mask":     spaces.Box(0.0, 1.0, shape=(MAX_HAND_SIZE,),                    dtype=np.float32),
            "table":         spaces.Box(0.0, 1.0, shape=(MAX_TABLE_SIZE, N_SUITS, N_RANKS), dtype=np.float32),
            "table_mask":    spaces.Box(0.0, 1.0, shape=(MAX_TABLE_SIZE,),                  dtype=np.float32),
            "prev_card":     spaces.Box(0.0, 1.0, shape=(N_SUITS, N_RANKS),                 dtype=np.float32),
            "prev_captured": spaces.Box(0.0, 1.0, shape=(1,),                               dtype=np.float32),
            "prev_scopa":    spaces.Box(0.0, 1.0, shape=(1,),                               dtype=np.float32),
            "current_player":spaces.Box(0.0, 1.0, shape=(2,),                               dtype=np.float32),
            "episode":       spaces.Box(0.0, 1.0, shape=(2,),                               dtype=np.float32),
            "remaining_deck":spaces.Box(0.0, 1.0, shape=(1,),                               dtype=np.float32),
            "score_stats":   spaces.Box(0.0, 1.0, shape=(8,),                               dtype=np.float32),
        })

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng.seed(seed)
        self.game = ScoponeGame2P()
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int):
        # Map "invalid" action indices to the first valid card
        valid_indices = list(range(len(self._current_player().hand)))
        if not valid_indices:
            raise RuntimeError("No cards in hand but step() called")

        if action not in valid_indices:
            action = valid_indices[0]

        step_result = self.game.step(action)
        done = step_result["done"]
        info = step_result["info"]

        # The reward dict is keyed by player_id.
        # We want the reward for the player who just moved.
        if not done:
            moved_player_id = 1 - self.game.current_player
        else:
            # At game end, reward dict already has final rewards for both;
            # we can pick the last_move_player as "moved player".
            moved_player_id = self.game.last_move_player if self.game.last_move_player is not None else 0

        reward = step_result["rewards"][moved_player_id]

        terminated = done
        truncated = False

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _current_player(self):
        return self.game.players[self.game.current_player]

    def _get_obs(self):
        """
        Build a human-like observation dict.
        """
        p = self._current_player()
        other = self.game.players[1 - self.game.current_player]

        # --- Hand: [9,4,10] + mask[9]
        hand_tensor = np.zeros((MAX_HAND_SIZE, N_SUITS, N_RANKS), dtype=np.float32)
        hand_mask = np.zeros(MAX_HAND_SIZE, dtype=np.float32)
        for i, card in enumerate(p.hand):
            if i >= MAX_HAND_SIZE:
                break
            hand_tensor[i, card.suit, card.rank - 1] = 1.0
            hand_mask[i] = 1.0

        # --- Table: [10,4,10] + mask[10]
        table_tensor = np.zeros((MAX_TABLE_SIZE, N_SUITS, N_RANKS), dtype=np.float32)
        table_mask = np.zeros(MAX_TABLE_SIZE, dtype=np.float32)
        for i, card in enumerate(self.game.table):
            if i >= MAX_TABLE_SIZE:
                break
            table_tensor[i, card.suit, card.rank - 1] = 1.0
            table_mask[i] = 1.0

        # --- Previous move: last played card, captured, scopa
        prev_card = np.zeros((N_SUITS, N_RANKS), dtype=np.float32)
        if self.game.last_played_card is not None:
            c = self.game.last_played_card
            prev_card[c.suit, c.rank - 1] = 1.0

        prev_captured = np.array(
            [1.0 if self.game.last_move_captured else 0.0],
            dtype=np.float32
        )
        prev_scopa = np.array(
            [1.0 if self.game.last_move_was_scopa else 0.0],
            dtype=np.float32
        )

        # --- Context: current player, episode, deck size
        current_player_onehot = np.zeros(2, dtype=np.float32)
        current_player_onehot[self.game.current_player] = 1.0

        ep_onehot = np.zeros(2, dtype=np.float32)
        ep_idx = min(self.game.episode_index, 1)
        ep_onehot[ep_idx] = 1.0

        remaining_deck = np.array(
            [len(self.game.deck) / 40.0],
            dtype=np.float32
        )

        # --- Score-like stats from piles (my vs opponent)
        def count_stats(cards):
            coins = sum(1 for c in cards if c.suit == Suit.COINS)
            sevens = sum(1 for c in cards if c.rank == 7)
            total = len(cards)
            return coins, sevens, total

        own_coins, own_sevens, own_total = count_stats(p.captured)
        opp_coins, opp_sevens, opp_total = count_stats(other.captured)
        own_scopas = p.scopas
        opp_scopas = other.scopas

        score_stats = np.array([
            own_coins / 10.0,
            own_sevens / 4.0,
            own_total / 40.0,
            own_scopas / 10.0,
            opp_coins / 10.0,
            opp_sevens / 4.0,
            opp_total / 40.0,
            opp_scopas / 10.0,
        ], dtype=np.float32)

        obs = {
            "hand":           hand_tensor,
            "hand_mask":      hand_mask,
            "table":          table_tensor,
            "table_mask":     table_mask,
            "prev_card":      prev_card,
            "prev_captured":  prev_captured,
            "prev_scopa":     prev_scopa,
            "current_player": current_player_onehot,
            "episode":        ep_onehot,
            "remaining_deck": remaining_deck,
            "score_stats":    score_stats,
        }
        return obs

    def render(self):
        p = self._current_player()
        print(f"\n--- Player {p.player_id}'s turn ---")
        print("Hand:")
        for i, c in enumerate(p.hand):
            print(f"  [{i}] {c}")
        print("Table:")
        for c in self.game.table:
            print(f"  {c}")
        for pl in self.game.players:
            print(f"Player {pl.player_id} captured ({pl.scopas} scopas):")
            for c in pl.captured:
                print(f"  {c}")
