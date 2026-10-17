"""README numbers and tables, read from results/ so no figure in the README is typed by hand.

``docs/README.template.md`` holds the prose with ``{{key|fmt}}`` placeholders and ``{{table:name}}`` blocks;
``scripts/build_readme.py`` fills them from the JSON files under ``results/``.  A key that cannot be resolved is an
error (never a silent blank), so the README cannot drift from the measurements.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PLACEHOLDER = re.compile(r"\{\{\s*([^{}|]+?)\s*(?:\|\s*([a-z0-9]+)\s*)?\}\}")


def fmt(value: Any, spec: str | None) -> str:
    """Format a number: pct0/pct1/pct2 (fraction -> percent), f1/f2/f3, int (thousands separators), sci1, h1, x1, str."""
    if spec in (None, "", "str"):
        return str(value)
    v = float(value)
    if spec.startswith("pct"):
        return f"{100 * v:.{int(spec[3:])}f}%"
    if spec.startswith("f"):
        return f"{v:.{int(spec[1:])}f}"  # f0 .. f3
    if spec == "int":
        return f"{int(round(v)):,}"
    if spec == "sci1":
        m, e = f"{v:.1e}".split("e")
        return f"{m}×10^{int(e)}"
    if spec == "h1":
        return f"{v:.1f} h"
    if spec == "x1":
        return f"{v:.1f}×"
    raise ValueError(f"unknown format {spec!r}")


def substitute(template: str, numbers: dict[str, Any], tables: dict[str, str]) -> str:
    """Replace ``{{key|fmt}}`` by the formatted number and ``{{table:name}}`` by the table; unknown keys raise KeyError."""

    def rep(m: re.Match) -> str:
        key, spec = m.group(1), m.group(2)
        if key.startswith("table:"):
            return tables[key[6:]]
        if key not in numbers:
            raise KeyError(f"README placeholder {key!r} has no value in results/")
        return fmt(numbers[key], spec)

    return PLACEHOLDER.sub(rep, template)


def _read(root: Path, rel: str) -> dict | None:
    p = root / rel
    return json.loads(p.read_text()) if p.exists() else None


def load_numbers(root: Path) -> dict[str, Any]:
    """Flat dict of every README number that exists under ``root/results`` (missing files are simply absent)."""
    N: dict[str, Any] = {}
    r = _read(root, "results/repro/repro250_summary.json")
    if r:
        mo, ma = r["metrics_ours"], r["metrics_asset_same_pixels"]
        N.update({
            "repro.n_pixels": r["n_pixels"], "repro.agreement": r["agreement"], "repro.interior": r["agreement_interior"],
            "repro.frac_interior": r["frac_interior"], "repro.boundary": r["agreement_boundary"], "repro.spearman": r["spearman"],
            "repro.within_one": r["within_one_level"], "repro.sb_ours": mo["basin_entropy"], "repro.sb_asset": ma["basin_entropy"],
            "repro.sbb_ours": mo["boundary_basin_entropy"], "repro.sbb_asset": ma["boundary_basin_entropy"],
            "repro.d_ours": mo["boundary_dimension"], "repro.d_asset": ma["boundary_dimension"],
            "repro.alpha_ours": mo["uncertainty_exponent"], "repro.alpha_asset": ma["uncertainty_exponent"],
            "repro.frac_solved": r["frac_solved"], "repro.frac_censored": r["frac_censored"],
        })
        up = (r.get("metrics_asset_full") or {}).get("1")
        if up:
            N.update({"up.sb": up["basin_entropy"], "up.sbb": up["boundary_basin_entropy"], "up.alpha": up["uncertainty_exponent"],
                      "up.d": up["boundary_dimension"], "up.boxes": up["boundary_box_fraction"], "up.side": up["shape"][0]})
    s = _read(root, "results/alice/stock_5246207.json")
    if s:
        N.update({"stock.fp32_agree": s["fp32:128"]["asset_agreement"], "stock.fp32_interior": s["fp32:128"]["asset_agreement_interior"],
                  "stock.fp32_boundary": s["fp32:128"]["asset_agreement_boundary"], "stock.vs_ours": s["fp32:128"]["vs_our_fast_fp32_settle_equal"],
                  "stock.tf32_agree": s["tf32:128"]["asset_agreement"], "stock.n": s["n"]})
    b = _read(root, "results/alice/smoke_5246129_bench.json")
    if b:
        ag, th = b["agreement"], b["throughput"]
        N.update({"smoke.n": b["n_sample_pixels"], "smoke.fp32_agree": ag["fp32"]["asset_agreement"], "smoke.fp32_interior": ag["fp32"]["asset_agreement_interior"],
                  "smoke.fp32_boundary": ag["fp32"]["asset_agreement_boundary"], "smoke.tf32_agree": ag["tf32"]["asset_agreement"],
                  "smoke.bf16_agree": ag["bf16"]["asset_agreement"], "smoke.tf32_interior": ag["tf32"]["asset_agreement_interior"],
                  "smoke.fp32_rps": th["fp32"]["B1024"]["rollouts_per_s"], "smoke.tf32_rps": th["tf32"]["B1024"]["rollouts_per_s"],
                  "smoke.bf16_rps": th["bf16"]["B1024"]["rollouts_per_s"], "smoke.gpu": b["env"]["gpu"], "smoke.torch": b["env"]["torch"]})
    e = _read(root, "results/alice/eechk_5246309.json")
    if e:
        N.update({"ee.settle_equal": e["pairs"]["full_B1024_a__early_B1024"]["settle_equal"], "ee.mean_loops": e["early_B1024"]["mean_loops"],
                  "ee.speedup": e["early_B1024"]["rollouts_per_s"] / e["full_B1024_a"]["rollouts_per_s"],
                  "ee.batch_invariant": e["pairs"]["full_B1024_a__full_B1000"]["settle_equal"],
                  "ee.b2048_equal": e["pairs"]["full_B1024_a__early_B2048"]["settle_equal"],
                  "ee.mig_full_rps": e["full_B1024_a"]["rollouts_per_s"], "ee.mig_early_rps": e["early_B1024"]["rollouts_per_s"]})
    t = _read(root, "results/throughput_m4.json")
    if t:
        best = {}
        for p in t["points"]:
            best[p["label"]] = max(best.get(p["label"], 0.0), p["rollouts_per_s"])
        N.update({"m4.mlx_rps": best["mlx"], "m4.mps_rps": best["torchfn-mps"], "m4.stock_mps_rps": best["stock-loopscape-mps"],
                  "m4.cpu_rps": best["torchfn-cpu"], "m4.gflop_per_rollout": t["flops_per_rollout_gflop"]})
    q = _read(root, "results/quickstart.json")
    if q:
        N.update({"quick.n": q["n_initial_conditions"], "quick.solved": q["n_solved"], "quick.seconds": q["seconds"]})
    m = _read(root, "results/metrics.json")
    if m and "hero" in m:
        h = m["hero"]
        N.update({"hero.pixels": h["pixels"], "hero.side": h["shape"][0], "hero.frac_solved": h["frac_solved"], "hero.frac_censored": h["frac_censored"],
                  "hero.n_outcomes": h["n_outcomes"], "hero.n_wrong_outcomes": h["n_wrong_outcomes"], "hero.mean_settle": h["mean_settle"],
                  "hero.max_settle": h["max_settle"], "hero.gpu_hours": (h.get("gpu_seconds") or 0) / 3600.0, "hero.mean_loops": h.get("mean_loops_computed"),
                  "hero.n_unsolved": round((1 - h["frac_solved"]) * h["pixels"]),
                  "hero.rps": h["pixels"] / h["gpu_seconds"] if h.get("gpu_seconds") else None})
        g = h.get("adjacent_gap") or {}
        for t, v in (g.get("frac_ge") or {}).items():
            N[f"hero.gap_ge{t}"] = v
        N["hero.max_gap"] = g.get("max_gap")
        for stride, mm in (h.get("settle_time_metrics_clipped") or {}).items():
            N.update({f"hero.c{stride}.sb": mm["basin_entropy"], f"hero.c{stride}.sbb": mm["boundary_basin_entropy"],
                      f"hero.c{stride}.alpha": mm["uncertainty_exponent"], f"hero.c{stride}.d": mm["boundary_dimension"],
                      f"hero.c{stride}.boxes": mm["boundary_box_fraction"], f"hero.c{stride}.shape": mm["shape"][0]})
        for stride, mm in (h.get("settle_time_metrics") or {}).items():
            N.update({f"hero.s{stride}.sb": mm["basin_entropy"], f"hero.s{stride}.sbb": mm["boundary_basin_entropy"],
                      f"hero.s{stride}.alpha": mm["uncertainty_exponent"], f"hero.s{stride}.d": mm["boundary_dimension"],
                      f"hero.s{stride}.boxes": mm["boundary_box_fraction"], f"hero.s{stride}.shape": mm["shape"][0]})
        kfs = m.get("keyframes") or []
        N["kf.n"] = len(kfs)
        N["kf.pixels"] = sum(k["pixels"] for k in kfs)
        N["kf.gpu_hours"] = sum((k.get("gpu_seconds") or 0) for k in kfs) / 3600.0
        if kfs:
            N["kf.zoom_max"] = max(k.get("zoom") or 0 for k in kfs)
    sd = _read(root, "results/scale_dependence.json")
    if sd:
        for r in sd["rows"]:
            m_ = re.match(r"keyframe x(\d+) ", r["label"])
            if m_:
                z = m_.group(1)
                N.update({f"scale.kf{z}.boxes": r["boundary_box_fraction"], f"scale.kf{z}.sb": r["basin_entropy"], f"scale.kf{z}.d": r["boundary_dimension"],
                          f"scale.kf{z}.pitch": r["pitch"]})
    gh = _read(root, "results/alice/gpu_hours.json")
    if gh:
        N.update({"gpu.jobs": gh["total"]["jobs"], "gpu.a100_hours": gh["total"]["a100_hours"], "gpu.raw_hours": gh["total"]["gpu_hours_raw"]})
    sc = _read(root, "results/scans.json")
    if sc:
        ps, lr = sc["pscan"], sc["longrun"]
        N.update({"pscan.puzzles": ps["n_puzzles"], "pscan.k": ps["k"], "pscan.starts": ps["n_puzzles"] * ps["k"], "pscan.frac_solved": ps["frac_solved"],
                  "pscan.always": ps["always_solved"], "pscan.never": ps["never_solved"], "pscan.partial": ps["partial"], "pscan.multistable": ps["multistable"],
                  "pscan.asset_frac": ps["asset"]["frac_solved"], "pscan.native_norm": ps.get("native_norm"), "pscan.plane_norm": ps.get("plane_corner_norm"), "long.n": lr["n"], "long.loops": lr["loops"], "long.frac_solved": lr["frac_solved_by_end"],
                  "long.changing": lr["still_wrong_and_changing"], "long.frozen": lr["still_wrong_and_frozen"], "long.median_first": lr["first_solved_percentiles"]["50"]})
        if sc.get("oscan"):
            N.update({f"oscan.{k}": v for k, v in sc["oscan"].items() if isinstance(v, (int, float))})
    rs = _read(root, "results/replay/summary.json")
    if rs and rs.get("roughness"):
        g = rs["roughness"]
        N["nudge.eps"] = rs["eps"]
        N["nudge.runs"] = rs["nudged"]
        for name, kinds in (("flat", ["rough0"]), ("edge", ["rough1"]), ("dust", ["rough2", "rough5"])):
            ks = [k for k in kinds if k in g]
            n = sum(g[k]["n"] for k in ks)
            if n:
                N[f"nudge.{name}_n"] = n
                N[f"plain.{name}"] = sum(g[k]["reproduced"] for k in ks) / n
                N[f"nudge.{name}"] = sum(g[k]["nudge_stable"] for k in ks) / n
                N[f"nudge.{name}_change"] = sum(g[k]["mean_max_change"] * g[k]["n"] for k in ks) / n
        N["plain.all"] = sum(v["reproduced"] for v in g.values()) / sum(v["n"] for v in g.values())
        bk = rs.get("by_kind", {})
        if "hero:slow" in bk:
            N["plain.slow_n"], N["plain.slow_ok"] = bk["hero:slow"]["items"], bk["hero:slow"]["reproduced"]
        N["plain.n"] = sum(v["n"] for v in g.values())
    ra = _read(root, "results/replay_agreement.json")
    if ra:
        tot: dict[str, list[int]] = {}
        for f in ra["fields"]:
            for row in f["by_roughness"]:
                a = tot.setdefault(row["roughness"], [0, 0, 0, 0])
                a[0] += row["n"]; a[1] += row["equal"]; a[2] += row["n_slow"]; a[3] += row["slow_equal"]
        grp = {"flat": ["0"], "edge": ["1"], "dust": ["2-4", "5-+"]}
        for name, keys in grp.items():
            n = sum(tot.get(k, [0] * 4)[0] for k in keys)
            eq = sum(tot.get(k, [0] * 4)[1] for k in keys)
            if n:
                N[f"replay.{name}"] = eq / n
                N[f"replay.{name}_n"] = n
        ns = sum(v[2] for v in tot.values())
        N["replay.slow_n"] = ns
        N["replay.slow_equal"] = sum(v[3] for v in tot.values())
        N["replay.n_per_field"] = ra["n_per_field"]
        N["replay.n_fields"] = len(ra["fields"])
    return N


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def md_table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join((align or "---") for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def tables(root: Path, N: dict[str, Any]) -> dict[str, str]:
    T: dict[str, str] = {}
    if "repro.agreement" in N:
        T["repro"] = md_table(
            ["", "ours (fp32, A100)", "upstream image"],
            [["pixels identical", f"**{fmt(N['repro.agreement'], 'pct1')}** of {fmt(N['repro.n_pixels'], 'int')}", ""],
             ["...where the upstream image is locally flat", f"{fmt(N['repro.interior'], 'pct2')} (on {fmt(N['repro.frac_interior'], 'pct0')} of the pixels)", ""],
             ["...inside the filaments", fmt(N["repro.boundary"], "pct1"), ""],
             ["Spearman correlation of settle time", fmt(N["repro.spearman"], "f3"), ""],
             ["basin entropy S_b (5x5 boxes)", fmt(N["repro.sb_ours"], "f3"), fmt(N["repro.sb_asset"], "f3")],
             ["boundary basin entropy S_bb", fmt(N["repro.sbb_ours"], "f3"), fmt(N["repro.sbb_asset"], "f3")],
             ["uncertainty exponent alpha", fmt(N["repro.alpha_ours"], "f3"), fmt(N["repro.alpha_asset"], "f3")],
             ["boundary dimension 2 - alpha", fmt(N["repro.d_ours"], "f3"), fmt(N["repro.d_asset"], "f3")]])
    sd = _read(root, "results/scale_dependence.json")
    if sd:
        rows = []
        for r in sd["rows"]:
            rows.append([r["label"], fmt(r["pitch"], "sci1"), "x".join(str(v) for v in r["shape"]), fmt(r["boundary_box_fraction"], "pct0"),
                         fmt(r["basin_entropy"], "f2"), fmt(r["boundary_dimension"], "f3")])
        T["scale"] = md_table(["field", "pixel pitch (plane units)", "pixels", "5x5 boxes touching a boundary", "S_b", "boundary dim."], rows)
    ce = _read(root, "results/cost_estimate.json")
    if ce:
        names = {"m4_mlx_fp32": "Apple M4, MLX", "a100_fp32_eager": "A100, fp32 eager", "a100_fp32_eager_early_exit": "A100, fp32 + early exit",
                 "a100_mig4g_fp32_full_early_exit": "A100 MIG 4g.40gb + early exit"}
        header = ["render"] + list(names.values())
        rows = []
        for label, est in ce["estimates"].items():
            rows.append([f"{label}: {fmt(est['pixels'], 'int')} px"] + [fmt(est["hours"][k]["hours"], "h1") for k in names])
        T["cost"] = md_table(header, rows)
    T.update(tables_results(root, N))
    return T


def _t(N: dict[str, Any], key: str, spec: str, default: str = "n/a") -> str:
    return fmt(N[key], spec) if N.get(key) is not None else default


def tables_results(root: Path, N: dict[str, Any]) -> dict[str, str]:
    """Tables that need the production results (hero metrics, replay agreement, scans, throughput)."""
    T: dict[str, str] = {}
    m = _read(root, "results/metrics.json")
    if m and "hero" in m:
        rows = [["starting thoughts (pixels)", f"{fmt(N['hero.pixels'], 'int')} ({N['hero.side']} x {N['hero.side']} over [-1, 1]^2)"],
                ["end on the true solution", f"{fmt(N['hero.frac_solved'], 'pct2')} ({fmt(N['hero.pixels'] - N['hero.n_unsolved'], 'int')})"],
                ["still changing at loop 24", f"{fmt(N['hero.frac_censored'], 'pct2')}"],
                ["distinct final grids (wrong ones)", f"{fmt(N['hero.n_outcomes'], 'int')} ({fmt(N['hero.n_wrong_outcomes'], 'int')})"],
                ["settle time, mean / max (loops)", f"{fmt(N['hero.mean_settle'], 'f2')} / {int(N['hero.max_settle'])}"],
                ["loops computed per pixel (early exit)", _t(N, "hero.mean_loops", "f1")]]
        for t in ("1", "2", "5", "10"):
            if N.get(f"hero.gap_ge{t}") is not None:
                rows.append([f"neighbour pairs differing by >= {t} loop{'s' if t != '1' else ''}", fmt(N[f"hero.gap_ge{t}"], "pct1")])
        rows.append(["largest neighbour gap (loops)", _t(N, "hero.max_gap", "int")])
        T["hero"] = md_table(["hero (full plane)", ""], rows)
        T["hero"] += "\n\n" + md_table(
            ["settle-time labels, 5x5 boxes", "pixel pitch", "boxes touching a boundary", "S_b", "S_bb", "alpha", "boundary dim. 2 - alpha"],
            [[f"hero, every {ordinal(int(st))} pixel" if st != "1" else "hero, every pixel", fmt(2.0 / (N["hero.side"] - 1) * int(st), "sci1"),
              fmt(N[f"hero.s{st}.boxes"], "pct0"), fmt(N[f"hero.s{st}.sb"], "f3"), fmt(N[f"hero.s{st}.sbb"], "f3"), fmt(N[f"hero.s{st}.alpha"], "f3"),
              fmt(N[f"hero.s{st}.d"], "f3")] for st in ("1", "2", "4") if f"hero.s{st}.sb" in N]
            + ([[f"hero, every {ordinal(int(st))} pixel, labels clipped to the upstream's 11 levels", fmt(2.0 / (N["hero.side"] - 1) * int(st), "sci1"),
                 fmt(N[f"hero.c{st}.boxes"], "pct0"), fmt(N[f"hero.c{st}.sb"], "f3"), fmt(N[f"hero.c{st}.sbb"], "f3"), fmt(N[f"hero.c{st}.alpha"], "f3"),
                 fmt(N[f"hero.c{st}.d"], "f3")] for st in ("2",) if f"hero.c{st}.sb" in N])
            + ([["upstream image, 1000 x 1000 (metrics computed by us from its 11 colour levels)", fmt(2.0 / (N["up.side"] - 1), "sci1"), fmt(N["up.boxes"], "pct0"),
                 fmt(N["up.sb"], "f3"), fmt(N["up.sbb"], "f3"), fmt(N["up.alpha"], "f3"), fmt(N["up.d"], "f3")]] if "up.sb" in N else []))
    ra = _read(root, "results/replay_agreement.json")
    if ra:
        tot: dict[str, list[int]] = {}
        for f in ra["fields"]:
            for row in f["by_roughness"]:
                a = tot.setdefault(row["roughness"], [0, 0, 0, 0])
                a[0] += row["n"]; a[1] += row["equal"]; a[2] += row["n_slow"]; a[3] += row["slow_equal"]
        names = {"0": "all 4 neighbours equal (flat)", "1": "a neighbour differs by 1 loop", "2-4": "a neighbour differs by 2-4 loops", "5-+": "a neighbour differs by 5 or more"}
        rows = [[names[k], fmt(v[0], "int"), fmt(v[1] / v[0], "pct0") if v[0] else "-", f"{v[3]} of {v[2]}" if v[2] else "-"] for k, v in
                ((k, tot[k]) for k in names if k in tot)]
        T["replay"] = md_table(["pixel's surroundings (stored field)", "pixels checked", "same settle time on a second float32 stack", "...of those stored as >= 8 loops"], rows)
    rs = _read(root, "results/replay/summary.json")
    if rs and rs.get("roughness"):
        g = rs["roughness"]
        names = {"rough0": "all 4 neighbours equal (flat)", "rough1": "a neighbour differs by 1 loop", "rough2": "a neighbour differs by 2-4 loops", "rough5": "a neighbour differs by 5 or more"}
        rows = [[names[k], fmt(v["n"], "int"), fmt(v["reproduced"] / v["n"], "pct1"), fmt(v["nudge_stable"] / v["n"], "pct0"), fmt(v["mean_max_change"], "f1")]
                for k, v in ((k, g[k]) for k in names if k in g)]
        T["replay_same"] = md_table(["pixel's surroundings (stored field)", "pixels re-run", "plain re-run reproduces the stored settle time",
                                     f"unchanged by all {rs['nudged']} nudged re-runs (relative {rs['eps']:g})", "mean largest change from a nudge (loops)"], rows)
    sc = _read(root, "results/scans.json")
    if sc:
        ps, lr = sc["pscan"], sc["longrun"]
        rows = [["puzzle scan", f"{fmt(ps['n_puzzles'], 'int')} puzzles x {ps['k']} EqR-style random starts, 24 loops (job {ps['job']})",
                 f"{fmt(ps['frac_solved'], 'pct1')} end on the solution; {fmt(ps['always_solved'], 'int')} puzzles always, {fmt(ps['never_solved'], 'int')} never, "
                 f"{fmt(ps['multistable'], 'int')} between 15% and 85%"],
                ["...the hero's puzzle", "same scan", f"{ps['asset']['frac_solved'] * ps['k']:.0f} of {ps['k']} starts solved (mean settle time {ps['asset']['mean_settle']:.1f})"],
                ["long run", f"{lr['n']} starts that were unsolved at loop 24, re-run for {lr['loops']} loops (job {lr['job']})",
                 f"{fmt(lr['frac_solved_by_end'], 'pct1')} reach the solution; {lr['still_wrong_and_changing']} are still changing, {lr['still_wrong_and_frozen']} are frozen on a wrong grid"]]
        if sc.get("oscan"):
            o = sc["oscan"]
            lo = min(v["frac_solved_min"] for v in o["puzzles"].values())
            rows.append(["outcome maps", f"{o['windows']} windows of 48 x 48 around native-scale base points, {len(o['puzzles'])} puzzles (job {o['job']}, cancelled at 15 of 32 windows)",
                         "; ".join(f"{'the hero puzzle' if k == 'asset' else 'puzzle #' + k[3:]}: {fmt(v['frac_solved_min'], 'pct1')}-{fmt(v['frac_solved_max'], 'pct1')} solved, "
                                    f"up to {fmt(v['distinct_final_max'], 'int')} distinct final grids"
                                    + (f", neighbouring pixels share their outcome {fmt(v['neighbour_agree_mean'], 'pct0')} of the time "
                                       f"(independent pixels would: {fmt(v['independent_agree_mean'], 'pct0')})" if v["frac_solved_min"] < 0.95 else "")
                                    for k, v in o["puzzles"].items())])
        T["scans"] = md_table(["scan", "what was run", "result"], rows)
    th = []
    if "m4.cpu_rps" in N:
        th += [["Apple M4 CPU, torch fp32", fmt(N["m4.cpu_rps"], "f2"), "results/throughput_m4.json"],
               ["Apple M4 GPU, torch MPS fp32", fmt(N["m4.mps_rps"], "f2"), "results/throughput_m4.json"],
               ["Apple M4 GPU, MLX fp32", fmt(N["m4.mlx_rps"], "f2"), "results/throughput_m4.json"],
               ["Apple M4 GPU, loopscape stock path (MPS)", fmt(N["m4.stock_mps_rps"], "f2"), "results/throughput_m4.json"]]
    if "smoke.fp32_rps" in N:
        th += [["A100-80GB, fp32 eager, batch 1024", fmt(N["smoke.fp32_rps"], "f1"), "results/alice/smoke_5246129_bench.json"],
               ["A100-80GB, TF32 (not used: different picture)", fmt(N["smoke.tf32_rps"], "f1"), "same"],
               ["A100-80GB, bfloat16 (not used)", fmt(N["smoke.bf16_rps"], "f1"), "same"]]
    if "ee.mig_full_rps" in N:
        th += [["A100 MIG 4g.40gb slice, fp32", fmt(N["ee.mig_full_rps"], "f1"), "results/alice/eechk_5246309.json"],
               ["A100 MIG 4g.40gb slice, fp32 + early exit", fmt(N["ee.mig_early_rps"], "f1"), "same"]]
    if N.get("hero.rps"):
        th += [["production hero tiles (mix of full A100s and MIG slices, early exit)", fmt(N["hero.rps"], "f1"), "results/metrics.json (pixels / summed tile seconds)"]]
    if th:
        T["throughput"] = md_table(["where", "rollouts per second (24 loops each)", "source"], th)
    return T
