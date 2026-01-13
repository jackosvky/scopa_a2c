from enum import IntEnum
from dataclasses import dataclass


class Suit(IntEnum):
    COINS = 0
    CUPS = 1
    SWORDS = 2
    CLUBS = 3


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: int  # 1..10  (8=jack, 9=queen, 10=king)

    @property
    def id(self) -> int:
        """Unique id 0..39."""
        return self.suit * 10 + (self.rank - 1)

    @staticmethod
    def from_id(cid: int) -> "Card":
        suit = Suit(cid // 10)
        rank = (cid % 10) + 1
        return Card(suit, rank)

    def __str__(self):
        return f"{self.rank} of {self.suit.name.lower()}"
