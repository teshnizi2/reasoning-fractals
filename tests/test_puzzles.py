"""The two puzzles that matter: the one behind the upstream asset, and the notebook's own."""
import hashlib

from render import puzzles as P
from render import sudoku as S


def test_asset_puzzle_hash_matches_the_upstream_asset_filename():
    # asset: basins_1000_0_eqr_sudoku_plane_rating47_7d68927f389fbdaa_rating47.png
    assert hashlib.blake2s(P.ASSET.question.encode(), digest_size=8).hexdigest() == "7d68927f389fbdaa"
    assert P.ASSET.rating == 47 and P.ASSET.source == "puzzles4_forum_hardest_1905"


def test_notebook_puzzle_is_a_different_harder_puzzle():
    assert P.NOTEBOOK.question != P.ASSET.question
    assert P.NOTEBOOK.rating == 162


def test_both_puzzles_have_unique_solutions_matching_the_dataset_answer():
    for p in (P.ASSET, P.NOTEBOOK):
        assert S.count_solutions(p.question, limit=2) == 1
        assert S.solve(p.question) == p.answer
        assert S.is_valid_solution(p.answer)
