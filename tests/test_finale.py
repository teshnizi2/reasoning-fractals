"""Pair finale of the video: two neighbouring starts ringed on the fractal, their Sudoku grids replaying beside them."""
import numpy as np

from render import finale as FI
from render import zoomvideo as ZV

Q = "2.6..4.9..7....6..8.......27.9.3..2..8.4.2......7.95..3.79...4.......3.6.1...3..."
A = "236184795974325618851697432769538124583412967142769583327956841498271356615843279"


VIEW = (0.001, 0.0, 100.0, 2.0)  # centre between the two starts; 24,000 screen px per plane unit at 480 px wide
LAY = dict(pair_y=190, card_size=170, card_y=330, gap=20)


def trace(settle: int) -> list[str]:
    """24 grids: blank-ish at first, the solution from loop ``settle`` on."""
    first = "".join("." if q == "." else q for q in Q)
    wrong = A[:30] + "9" + A[31:]
    return [first] * 1 + [wrong] * (settle - 1) + [A] * (24 - settle)


def spec():
    return dict(
        puzzle=Q, solution=A, t_in=1.0, loop_seconds=0.5, loops_shown=14, pitch=0.002,
        a=dict(x=0.0, y=0.0, settle=3, trace=trace(3), label="A"),
        b=dict(x=0.002, y=0.0, settle=11, trace=trace(11), label="B"),
    )


def test_loop_clock_waits_then_advances_one_loop_per_step_and_stops():
    s = spec()
    assert FI.loop_at(0.5, s) == 0
    assert FI.loop_at(s["t_in"] + FI.CARD_FADE, s) == 0  # cards are up, replay not started
    t0 = s["t_in"] + FI.CARD_FADE + FI.REPLAY_DELAY
    assert FI.loop_at(t0 + 0.01, s) == 1
    assert FI.loop_at(t0 + 0.01 + 0.5 * 3, s) == 4
    assert FI.loop_at(t0 + 1000.0, s) == s["loops_shown"]


def test_nothing_is_drawn_before_t_in():
    f = np.zeros((600, 480, 3), np.uint8)
    out = FI.draw_finale(f, spec(), 0.5, view=VIEW, layout=LAY)
    np.testing.assert_array_equal(out, f)


def test_rings_sit_on_the_blocks_and_cards_below():
    w, h = 480, 600
    f = np.zeros((h, w, 3), np.uint8)
    out = FI.draw_finale(f, spec(), 6.0, view=VIEW, layout=LAY)
    ax, ay = ZV.to_screen(0.0, 0.0, 0.001, 0.0, 100.0, (w, h), 2.0)
    # the ring around A is amber and surrounds (not covers) its block: pixel on the ring's edge is amber, the centre is not
    half = FI.block_px(VIEW, (w, h), spec()) / 2  # 48 px blocks
    edge = out[int(ay), int(ax - half - 2)]
    assert edge[0] > 200 and edge[1] > 120 and edge[2] < 80
    assert out[int(ay), int(ax)].sum() == 0
    # cards: something drawn in the card band, nothing at the very bottom
    assert out[330:520].sum() > 0 and out[580:].sum() == 0


def test_cards_show_the_loop_being_replayed():
    w, h = 480, 600
    f = np.zeros((h, w, 3), np.uint8)
    s = spec()
    t0 = s["t_in"] + FI.CARD_FADE + FI.REPLAY_DELAY
    early = FI.draw_finale(f, s, t0 + 0.1, view=VIEW, layout=LAY)
    late = FI.draw_finale(f, s, t0 + 8.0, view=VIEW, layout=LAY)
    assert np.abs(early.astype(int) - late.astype(int)).sum() > 0
