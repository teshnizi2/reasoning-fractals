"""Featured starts: representative pixels to show in the explorer (candidates only; replay is verified elsewhere)."""
import numpy as np

from render import featured as F


def field():
    s = np.full((40, 40), 4, dtype=np.uint8)
    s[:, 20:] = 2                    # a big flat domain of the fastest value on the right
    s[5, 5] = 15                     # a lone slow pixel
    s[10, 10], s[10, 11] = 1, 14     # neighbours that disagree strongly (gap 13 beats the lone pixel's gap 11)
    flags = np.ones((40, 40), dtype=np.uint8)
    flags[30, 3] = 0                 # one wrong ending
    flags[31, 31] = 1 | 4            # one unsettled pixel (solved but still changing at the cap)
    return s, flags


def test_fast_candidate_is_deep_inside_the_largest_flat_domain_of_the_fastest_value():
    s, f = field()
    cands = F.fast(s, f, k=3)
    r, c = cands[0]
    assert s[r, c] == 2 and c >= 25  # far from the domain edge at column 20


def test_slow_candidates_are_solved_uncensored_and_sorted_by_time():
    s, f = field()
    cands = F.slow(s, f, k=2)
    assert cands[0] == (5, 5)


def test_pair_candidates_are_adjacent_solved_pixels_with_a_large_gap():
    s, f = field()
    (r, c), = F.pair(s, f, k=1, min_gap=6)
    assert (r, c) == (10, 10) and abs(int(s[r, c]) - int(s[r, c + 1])) >= 6


def test_wrong_candidates_prefer_settled_wrong_endings_then_unsettled_ones():
    s, f = field()
    assert F.wrong(s, f, k=1)[0] == (30, 3)
    f2 = f.copy()
    f2[30, 3] = 1
    assert F.wrong(s, f2, k=1)[0] == (31, 31)  # fall back to the unsettled pixel
    f3 = np.ones_like(f)
    assert F.wrong(s, f3, k=1) == []


def test_to_display_flips_rows_so_that_row_zero_is_the_top():
    assert F.to_display(0, 7, ny=100) == (99, 7)
    assert F.to_display(99, 7, ny=100) == (0, 7)
