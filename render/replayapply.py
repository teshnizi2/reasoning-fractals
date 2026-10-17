"""Turn replay results into the explorer's featured starts and the video's finale pair.

Inputs are the planner's items (``replayplan.plan_layer``) and the cluster's results (``render.replay.run_items``),
matched by id.  A candidate is only used if its plain replay reproduced the stored settle time and outcome.
"""
from __future__ import annotations

from collections import defaultdict


def _groups(plan: list[dict], results: list[dict]) -> dict[tuple[str, str, int], dict[str, tuple[dict, dict]]]:
    by_id = {r["id"]: r for r in results}
    out: dict[tuple[str, str, int], dict[str, tuple[dict, dict]]] = defaultdict(dict)
    for p in plan:
        if p["id"] in by_id:
            out[(p["layer"], p["kind"], p["group"])][p["role"]] = (p, by_id[p["id"]])
    return out


def sensitivity_note(settle: int, nudged: list[int], unsettled: bool = False) -> str:
    n = len(nudged)
    here = "still changing at loop 24" if unsettled else str(settle)
    if all(v == settle for v in nudged):
        return f"{n} re-runs of this start with a rounding-sized nudge (relative 1e-6) all agree: {here} loops." if not unsettled else \
            f"{n} re-runs of this start with a rounding-sized nudge (relative 1e-6) all end the same way."
    return (f"Rounding-sensitive: {n} re-runs with a nudge of relative 1e-6 give {', '.join(str(v) for v in nudged)} loops "
            f"(this computation: {here}).")


def featured(plan: list[dict], results: list[dict], ny: int) -> list[dict]:
    """First verified candidate of each kind (fast, slow, pair, wrong), in display coordinates of the hero (layer 0)."""
    groups = _groups(plan, results)
    out = []
    spec = {
        "fast": ("a quick thought", 6, lambda T, g: f"settles in {T} loops"),
        "slow": ("a slow thought", 24, lambda T, g: f"takes {T} loops"),
        "pair": ("neighbours that disagree", 40, lambda T, g: f"next to a pixel {g} loops away"),
        "wrong": ("a wrong ending", 24, lambda T, g: "ends on a wrong grid"),
    }
    for kind in ("fast", "slow", "pair", "wrong"):
        label, zoom, title = spec[kind]
        for key in sorted(k for k in groups if k[0] == "hero" and k[1] == kind):
            g = groups[key]
            if kind == "pair":
                if "a" not in g or "b" not in g or not (g["a"][1]["equal"] and g["b"][1]["equal"]):
                    continue
                gap = abs(g["b"][1]["settle"] - g["a"][1]["settle"])
                if gap < 4:
                    continue
                p, r = g["a"]
                tt = title(r["settle"], gap)
            else:
                if "single" not in g or not g["single"][1]["equal"]:
                    continue
                p, r = g["single"]
                tt = title(r["settle"], r["settle"])
            out.append(dict(id=kind, label=label, title=tt, layer=0, col=int(p["col"]), row=int(ny - 1 - p["row"]), zoom=zoom,
                            settle=int(r["settle"]), solved=bool(r["flags"] & 1), trace=r["trace"],
                            note=sensitivity_note(int(r["settle"]), r["nudged"], unsettled=bool(r["flags"] & 4))))
            break
    return out


def _monotone(trace: list[str], solution: str, settle: int) -> bool:
    eq = [t == solution for t in trace]
    return not any(eq[:settle]) and all(eq[settle:])


def pick_finale(plan: list[dict], results: list[dict], pitches: dict[str, float], solution: str, min_robust_gap: int = 3,
                criterion: str = "strict", rank: str = "robust") -> dict | None:
    """Best neighbouring pair of a keyframe for the video: both verified, monotone, and the slow/fast ordering survives
    the rounding-sized nudges.  ``strict``: every re-run keeps a gap of at least ``min_robust_gap`` loops; ``median``: the
    median gap over the plain and nudged runs is at least that and the order holds in all but one run.  Ranked by the
    robust (or median) gap, then depth, then stored gap; with ``rank="depth"`` the finest layer that passes wins (then the robust gap)."""
    groups = _groups(plan, results)
    best = None
    for (layer, kind, gi), g in groups.items():
        if kind != "pair" or not layer.startswith("kf") or "a" not in g or "b" not in g:
            continue
        (pa, ra), (pb, rb) = g["a"], g["b"]
        if not (ra["equal"] and rb["equal"]):
            continue
        ta, tb = ra["settle"], rb["settle"]
        if ta == tb or not (ra["flags"] & 1 and rb["flags"] & 1) or (ra["flags"] | rb["flags"]) & 4:
            continue
        if not (_monotone(ra["trace"], solution, ta) and _monotone(rb["trace"], solution, tb)):
            continue
        sgn = 1 if tb > ta else -1
        gaps = [sgn * (tb - ta)] + [sgn * (nb - na) for na, nb in zip(ra["nudged"], rb["nudged"])]
        keeping = sum(1 for g in gaps if g > 0)
        if criterion == "strict":
            robust = min(gaps)
        elif criterion == "median":
            robust = sorted(gaps)[len(gaps) // 2]
            if keeping < len(gaps) - 1:
                continue
        else:
            raise ValueError(f"unknown criterion {criterion!r}")
        if robust < min_robust_gap:
            continue
        score = (robust, -pitches[layer], abs(tb - ta)) if rank == "robust" else (-pitches[layer], robust, abs(tb - ta))
        if best is None or score > best[0]:
            best = (score, dict(
                layer=layer, gap=abs(tb - ta), pitch=pitches[layer], render_row=pa["row"], col=pa["col"],
                a=dict(x=pa["x"], y=pa["y"], settle=ta, trace=ra["trace"], label="A"),
                b=dict(x=pb["x"], y=pb["y"], settle=tb, trace=rb["trace"], label="B"),
                verified=dict(criterion=criterion, rank=rank, robust_gap=robust, runs_keeping_order=keeping, runs=len(gaps), a_range=[min([ta] + ra["nudged"]), max([ta] + ra["nudged"])],
                              b_range=[min([tb] + rb["nudged"]), max([tb] + rb["nudged"])], nudged_runs=len(ra["nudged"])),
            ))
    return best[1] if best else None


def roughness_summary(plan: list[dict], results: list[dict]) -> dict[str, dict]:
    """Per roughness stratum (kinds ``rough*``): pixels replayed, how many the plain replay reproduced, how many were unchanged
    by every nudged re-run, and the mean (over pixels) of the largest settle-time change a nudge caused."""
    by_id = {r["id"]: r for r in results}
    acc: dict[str, list] = defaultdict(list)
    for p in plan:
        if p["kind"].startswith("rough") and p["id"] in by_id:
            acc[p["kind"]].append(by_id[p["id"]])
    out = {}
    for kind, rs in sorted(acc.items()):
        out[kind] = dict(n=len(rs), reproduced=sum(int(r["equal"]) for r in rs),
                         nudge_stable=sum(int(all(v == r["settle"] for v in r["nudged"])) for r in rs),
                         mean_max_change=sum(max(abs(v - r["settle"]) for v in r["nudged"]) for r in rs) / len(rs))
    return out
