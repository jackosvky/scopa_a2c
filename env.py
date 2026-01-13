import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

from .cards import Card, Suit
from .game import ScoponeGame2P


MAX_HAND_SIZE = 9
N_CARDS = 40


class ScoponeEnv(gym.Env):
    """
    Gymnasium env for 2-player Scopone.

    - Single-agent interface with self-play:
      the agent always controls "current_player" (0 or 1),
      and the same policy is used for both players.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self._rng = np.random.RandomState(seed)
        self.game = ScoponeGame2P()

        # Action: index of card in hand (0..MAX_HAND_SIZE-1)
        self.action_space = spaces.Discrete(MAX_HAND_SIZE)

        # Observation: flat vector
        # We include:
        # - card location masks: hand[40], table[40], own_pile[40], opp_pile[40]  => 160
        # - hand slot mask: 9
        # - current_player one-hot: 2
        # - episode_index: 2 one-hot
        # - remaining_deck_size: 1 (normalized)
        # - table_size: 1 (normalized)
        # - own/opp basic counts: coins, sevens, total_captured, scopas => 4 * 2 = 8
        # Total ~ 160 + 9 + 2 + 2 + 1 + 1 + 8 = 183
        obs_dim = 160 + 9 + 2 + 2 + 1 + 1 + 8
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

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
            # Should only happen right after round end, but guard anyway
            raise RuntimeError("No cards in hand but step() called")

        if action not in valid_indices:
            action = valid_indices[0]

        step_result = self.game.step(action)
        done = step_result["done"]
        info = step_result["info"]

        # Reward from current player's perspective:
        # After step(), game.current_player has already advanced if not done.
        # The reward dict in game.step() is keyed by player_id.
        # We want the reward for the player who just moved.
        moved_player_id = 1 - self.game.current_player if not done else (
            0 if info.get("score_team0") is not None else 0  # safe default
        )

        reward = step_result["rewards"][moved_player_id]

        # In episodic RL with Gymnasium, we usually use 'terminated' (done by game rules)
        # and 'truncated' (time limit). Here, no truncation logic yet.
        terminated = done
        truncated = False

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _current_player(self):
        return self.game.players[self.game.current_player]

    def _get_obs(self) -> np.ndarray:
        """
        Encode full public + own info into a flat vector.
        """
        p = self._current_player()
        other = self.game.players[1 - self.game.current_player]

        # Masks: 4 x 40
        hand_mask = np.zeros(N_CARDS, dtype=np.float32)
        table_mask = np.zeros(N_CARDS, dtype=np.float32)
        own_pile_mask = np.zeros(N_CARDS, dtype=np.float32)
        opp_pile_mask = np.zeros(N_CARDS, dtype=np.float32)

        for c in p.hand:
            hand_mask[c.id] = 1.0
        for c in self.game.table:
            table_mask[c.id] = 1.0
        for c in p.captured:
            own_pile_mask[c.id] = 1.0
        for c in other.captured:
            opp_pile_mask[c.id] = 1.0

        # Hand slot mask (max 9)
        hand_slot_mask = np.zeros(MAX_HAND_SIZE, dtype=np.float32)
        for i in range(len(p.hand)):
            hand_slot_mask[i] = 1.0

        # Current player one-hot
        cur_player_onehot = np.zeros(2, dtype=np.float32)
        cur_player_onehot[self.game.current_player] = 1.0

        # Episode index one-hot: 0 or 1 (2 episodes max)
        ep_onehot = np.zeros(2, dtype=np.float32)
        ep_idx = min(self.game.episode_index, 1)
        ep_onehot[ep_idx] = 1.0

        # Remaining deck size (normalized)
        remaining_deck = len(self.game.deck) / 40.0

        # Table size (normalized)
        table_size = len(self.game.table) / 40.0

        # Own/opp captured stats
        def count_stats(cards):
            coins = sum(1 for c in cards if c.suit == Suit.COINS)
            sevens = sum(1 for c in cards if c.rank == 7)
            total = len(cards)
            return coins, sevens, total

        own_coins, own_sevens, own_total = count_stats(p.captured)
        opp_coins, opp_sevens, opp_total = count_stats(other.captured)
        own_scopas = p.scopas
        opp_scopas = other.scopas

        stats = np.array([
            own_coins / 10.0,
            own_sevens / 4.0,
            own_total / 40.0,
            own_scopas / 10.0,
            opp_coins / 10.0,
            opp_sevens / 4.0,
            opp_total / 40.0,
            opp_scopas / 10.0,
        ], dtype=np.float32)

        obs = np.concatenate([
            hand_mask,
            table_mask,
            own_pile_mask,
            opp_pile_mask,
            hand_slot_mask,
            cur_player_onehot,
            ep_onehot,
            np.array([remaining_deck], dtype=np.float32),
            np.array([table_size], dtype=np.float32),
            stats,
        ])
        return obs.astype(np.float32)

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
