"""Choosing the neighbouring pair for the video finale from a rendered keyframe."""
import numpy as np

from render import pairpick as PP


def field(ny=60, nx=48):
    rng = np.random.default_rng(3)
    s = rng.integers(4, 9, size=(ny, nx)).astype(np.uint8)
    f = np.ones((ny, nx), np.uint8)  # bit 0: solved; bit 2 (censored) clear
    return s, f


def test_candidates_are_horizontally_adjacent_solved_pairs_with_a_big_gap_inside_the_margin():
    s, f = field()
    s[30, 20], s[30, 21] = 4, 15  # gap 11 near the centre
    s[3, 5], s[3, 6] = 4, 20      # bigger gap but inside the margin
    f[40, 10] = 0                 # unsolved pixel next to a slow one: not a candidate
    s[40, 10], s[40, 11] = 4, 18
    c = PP.candidates(s, f, min_gap=6, margin=8, k=5)
    assert (30, 20, 11) in c
    assert all(8 <= r < 60 - 8 and 8 <= col < 48 - 8 - 1 for r, col, _ in c)
    assert not any(r == 40 and col == 10 for r, col, _ in c)
    assert not any(r == 3 for r, _, _ in c)


def test_censored_pixels_are_never_candidates():
    s, f = field()
    s[30, 20], s[30, 21] = 4, 22
    f[30, 21] |= 4  # still changing at the last loop
    c = PP.candidates(s, f, min_gap=6, margin=8, k=50)
    assert not any(r == 30 and col in (20, 21) for r, col, _ in c)


def test_ranking_prefers_a_bigger_gap_then_the_centre():
    s, f = field()
    s[30, 24], s[30, 25] = 4, 14  # gap 10, centre
    s[12, 12], s[12, 13] = 4, 14  # gap 10, off-centre
    s[20, 30], s[20, 31] = 4, 18  # gap 14
    c = PP.candidates(s, f, min_gap=6, margin=8, k=10)
    assert c[0][2] == 14 or c[0][:2] == (20, 30)
    order = [x[:2] for x in c]
    assert order.index((30, 24)) < order.index((12, 12))
