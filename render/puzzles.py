"""The Sudoku puzzles used in this project (Sudoku-Extreme test split, HF sapientinc/sudoku-extreme).

ASSET is the puzzle behind loopscape's 1000x1000 EqR basin image: the file name carries
``rating47_7d68927f389fbdaa`` and blake2s(question, digest_size=8) of this puzzle is exactly that hash.
NOTEBOOK is the puzzle in loopscape's demos/basins.ipynb "EqR on Sudoku" cell (a different, harder puzzle).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Puzzle:
    name: str
    question: str
    answer: str
    rating: int
    source: str


ASSET = Puzzle(
    name="asset",
    question="2.6..4.9..7....6..8.......27.9.3..2..8.4.2......7.95..3.79...4.......3.6.1...3...",
    answer="236184795974325618851697432769538124583412967142769583327956841498271356615843279",
    rating=47,
    source="puzzles4_forum_hardest_1905",
)

NOTEBOOK = Puzzle(
    name="notebook",
    question=".....6.....7.3.4..5..8.....9.8.7.3...1.....9...498.7....2....4387....2......2....",
    answer="349756182687132459521894637958671324716243598234985716192567843873419265465328971",
    rating=162,
    source="puzzles4_forum_hardest_1905",
)
