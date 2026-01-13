import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from .cards import Card, Suit


@dataclass
class PlayerState:
    player_id: int
    team_id: int
    hand: List[Card] = field(default_factory=list)
    captured: List[Card] = field(default_factory=list)
    scopas: int = 0


class ScoponeGame2P:
    """
    2-player Scopone (Val Seriana variant) engine.

    - 2 players, 2 teams (player0=team0, player1=team1)
    - Deck of 40 cards, 4 suits x 10 ranks.
    - Deal 9+9 and 4 to table, then another 9+9 from deck (two episodes).
    - Single-card capture priority.
    - Ace clears table but never Scopa.
    - Scopa: clear table with non-ace, except last card of game (handled at end).
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.players: List[PlayerState] = []
        self.table: List[Card] = []
        self.deck: List[Card] = []
        self.current_player: int = 0
        self.last_capture_player: Optional[int] = None
        self.episode_index: int = 0  # 0 or 1
        self.done: bool = False

        # For correct "no last-card Scopa" handling:
        self.last_move_was_scopa: bool = False
        self.last_move_player: Optional[int] = None

        self.reset()

    # ------------------------------------------------------------------ #
    # Setup / dealing
    # ------------------------------------------------------------------ #

    def reset(self):
        self.deck = [Card(suit, rank)
                     for suit in Suit
                     for rank in range(1, 11)]
        self.rng.shuffle(self.deck)

        self.players = [
            PlayerState(player_id=0, team_id=0),
            PlayerState(player_id=1, team_id=1),
        ]
        self.table = []
        self.current_player = 0
        self.last_capture_player = None
        self.episode_index = 0
        self.done = False
        self.last_move_was_scopa = False
        self.last_move_player = None

        self._deal_initial()

    def _deal_initial(self):
        # Clear hands and table
        for p in self.players:
            p.hand.clear()
            p.captured.clear()
            p.scopas = 0
        self.table.clear()

        # Deal 9 cards each, then 4 on table
        for _ in range(9):
            self.players[0].hand.append(self.deck.pop())
            self.players[1].hand.append(self.deck.pop())
        for _ in range(4):
            self.table.append(self.deck.pop())

    def _deal_next_episode(self):
        """In 2-player version we deal a second 'episode' of 9+9."""
        for _ in range(9):
            self.players[0].hand.append(self.deck.pop())
            self.players[1].hand.append(self.deck.pop())
        self.episode_index += 1

    # ------------------------------------------------------------------ #
    # Core game logic
    # ------------------------------------------------------------------ #

    def is_round_over(self) -> bool:
        """Hands empty for both players."""
        return all(len(p.hand) == 0 for p in self.players)

    def step(self, action_card_index: int) -> Dict:
        """
        Execute one move by current_player.
        :param action_card_index: index in that player's hand (0..len(hand)-1).
        :return: dict with:
            - 'rewards': {player_id: float}
            - 'done': bool
            - 'info': dict
        """
        if self.done:
            raise RuntimeError("Game is already finished")

        player = self.players[self.current_player]
        assert 0 <= action_card_index < len(player.hand)

        played_card = player.hand.pop(action_card_index)
        captured: List[Card] = []
        scopa_made = False

        # Ace rule: always clear table, no Scopa
        if played_card.rank == 1:
            if self.table:
                captured = self.table[:] + [played_card]
                self.table.clear()
            else:
                # Ace to empty table -> just 'play' it
                self.table.append(played_card)

        else:
            # Single-card priority
            equal_indices = [i for i, c in enumerate(self.table)
                             if c.rank == played_card.rank]
            if equal_indices:
                # Capture first equal-rank card
                chosen_idx = equal_indices[0]
                chosen_card = self.table.pop(chosen_idx)
                captured = [chosen_card, played_card]
            else:
                # Try sum combinations
                subset = self._choose_sum_subset(played_card)
                if subset:
                    for c in subset:
                        self.table.remove(c)
                    captured = subset + [played_card]
                else:
                    # No capture
                    self.table.append(played_card)

        if captured:
            # Move to player pile
            player.captured.extend(captured)
            self.last_capture_player = self.current_player

            # Check Scopa: table now empty, played non-ace,
            # but we don't yet know if this was the last card of game.
            if played_card.rank != 1 and len(self.table) == 0:
                scopa_made = True
                player.scopas += 1

        # Bookkeeping for "last move Scopa removal"
        self.last_move_was_scopa = scopa_made
        self.last_move_player = self.current_player

        reward_dict = {p.player_id: 0.0 for p in self.players}
        info: Dict = {}

        # Round ended?
        if self.is_round_over():
            if len(self.deck) > 0:
                # Second episode
                self._deal_next_episode()
            else:
                # Entire game finished
                self._finish_game(reward_dict, info)
                self.done = True

        if not self.done:
            self.current_player = 1 - self.current_player

        info["current_player"] = self.current_player
        info["episode_index"] = self.episode_index
        info["table_size"] = len(self.table)

        return {
            "rewards": reward_dict,
            "done": self.done,
            "info": info,
        }

    # ------------------------------------------------------------------ #
    # Sum-combination selection (deterministic heuristic, Phase 1)
    # ------------------------------------------------------------------ #

    def _choose_sum_subset(self, played_card: Card) -> List[Card]:
        """
        Among all subsets of table cards whose summed ranks equal played_card.rank,
        select one heuristically:
        - maximize #coins captured
        - tie-break by subset size
        - then by lexicographic card-id order
        """
        target = played_card.rank
        table_cards = self.table
        n = len(table_cards)
        best_subset: List[Card] = []
        best_key: Optional[Tuple[int, int, Tuple[int, ...]]] = None

        for mask in range(1, 1 << n):
            subset: List[Card] = []
            s = 0
            coins_count = 0
            ids: List[int] = []
            for i in range(n):
                if mask & (1 << i):
                    c = table_cards[i]
                    subset.append(c)
                    s += c.rank
                    ids.append(c.id)
                    if c.suit == Suit.COINS:
                        coins_count += 1
            if s == target:
                key = (coins_count, len(subset), tuple(sorted(ids)))
                if best_key is None or key > best_key:
                    best_key = key
                    best_subset = subset

        return best_subset

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _finish_game(self, reward_dict: Dict[int, float], info: Dict):
        # Remaining table cards to last-capture player
        if self.last_capture_player is not None and self.table:
            self.players[self.last_capture_player].captured.extend(self.table)
            self.table.clear()

        # Remove Scopa if it came from the very last card of the game
        if self.last_move_was_scopa and self.last_move_player is not None:
            last_player = self.players[self.last_move_player]
            if last_player.scopas > 0:
                last_player.scopas -= 1  # remove last scopa

        # Aggregate by team (team0: player0, team1: player1)
        team_cards = {0: [], 1: []}
        team_scopa = {0: 0, 1: 0}
        for p in self.players:
            team_cards[p.team_id].extend(p.captured)
            team_scopa[p.team_id] += p.scopas

        score_team = {0: 0, 1: 0}

        # 1) Majority coins
        coins_count = {
            t: sum(1 for c in team_cards[t] if c.suit == Suit.COINS)
            for t in (0, 1)
        }
        info["coins_team0"] = coins_count[0]
        info["coins_team1"] = coins_count[1]
        if coins_count[0] > coins_count[1]:
            score_team[0] += 1
        elif coins_count[1] > coins_count[0]:
            score_team[1] += 1

        # 2) Coin sequence including 1,2,3
        def longest_coin_seq_including_1_2_3(cards: List[Card]) -> int:
            ranks = sorted({c.rank for c in cards if c.suit == Suit.COINS})
            if not ranks:
                return 0
            longest = 0
            cur_len = 0
            start = None
            last = None
            for r in ranks:
                if last is None or r == last + 1:
                    if cur_len == 0:
                        start = r
                    cur_len += 1
                else:
                    if start is not None and last is not None:
                        if start <= 1 <= last and start <= 2 <= last and start <= 3 <= last:
                            longest = max(longest, cur_len)
                    cur_len = 1
                    start = r
                last = r
            if cur_len > 0 and start is not None and last is not None:
                if start <= 1 <= last and start <= 2 <= last and start <= 3 <= last:
                    longest = max(longest, cur_len)
            return longest

        seq0 = longest_coin_seq_including_1_2_3(team_cards[0])
        seq1 = longest_coin_seq_including_1_2_3(team_cards[1])
        info["coin_seq_team0"] = seq0
        info["coin_seq_team1"] = seq1
        if seq0 >= 3:
            score_team[0] += seq0
        if seq1 >= 3:
            score_team[1] += seq1

        # 3) Majority 7s
        sevens_count = {
            t: sum(1 for c in team_cards[t] if c.rank == 7)
            for t in (0, 1)
        }
        info["sevens_team0"] = sevens_count[0]
        info["sevens_team1"] = sevens_count[1]
        if sevens_count[0] > sevens_count[1]:
            score_team[0] += 1
        elif sevens_count[1] > sevens_count[0]:
            score_team[1] += 1

        # 4) Majority total cards
        total_cards = {t: len(team_cards[t]) for t in (0, 1)}
        info["total_cards_team0"] = total_cards[0]
        info["total_cards_team1"] = total_cards[1]
        if total_cards[0] > total_cards[1]:
            score_team[0] += 1
        elif total_cards[1] > total_cards[0]:
            score_team[1] += 1

        # 5) 7 of coins
        has_7coins = {
            t: any(c.suit == Suit.COINS and c.rank == 7 for c in team_cards[t])
            for t in (0, 1)
        }
        info["has_7coins_team0"] = has_7coins[0]
        info["has_7coins_team1"] = has_7coins[1]
        if has_7coins[0]:
            score_team[0] += 1
        if has_7coins[1]:
            score_team[1] += 1

        # 6) Scopa
        info["scopa_team0"] = team_scopa[0]
        info["scopa_team1"] = team_scopa[1]
        score_team[0] += team_scopa[0]
        score_team[1] += team_scopa[1]

        info["score_team0"] = score_team[0]
        info["score_team1"] = score_team[1]

        # Final reward: team0_score - team1_score
        team_reward = score_team[0] - score_team[1]
        for p in self.players:
            if p.team_id == 0:
                reward_dict[p.player_id] = float(team_reward)
            else:
                reward_dict[p.player_id] = float(-team_reward)
