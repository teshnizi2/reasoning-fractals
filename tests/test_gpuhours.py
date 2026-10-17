"""GPU-hour accounting from sacct rows (A100 MIG slices count as fractions of an A100)."""
from render import gpuhours as G


def test_parse_alloc_tres_reads_gpu_types():
    d = G.parse_alloc_tres("billing=141,cpu=4,gres/gpu=1,gres/gpu:a100=1,mem=32G,node=1")
    assert d == {"a100": 1}
    d = G.parse_alloc_tres("cpu=4,gres/gpu=1,gres/gpu:4g.40gb=1,mem=24G")
    assert d == {"4g.40gb": 1}
    assert G.parse_alloc_tres("cpu=4,mem=8G") == {}


def test_a100_equivalent_weights_mig_slices():
    assert G.a100_equiv({"a100": 1}) == 1.0
    assert abs(G.a100_equiv({"4g.40gb": 1}) - 4 / 7) < 1e-12
    assert abs(G.a100_equiv({"3g.40gb": 2}) - 6 / 7) < 1e-12


def test_summarize_groups_by_job_name_and_sums_hours():
    rows = [
        dict(JobID="1_0", JobName="rf-render", ElapsedRaw="3600", AllocTRES="gres/gpu:a100=1", State="COMPLETED"),
        dict(JobID="1_1", JobName="rf-render", ElapsedRaw="1800", AllocTRES="gres/gpu:4g.40gb=1", State="COMPLETED"),
        dict(JobID="2", JobName="rf-pscan", ElapsedRaw="7200", AllocTRES="gres/gpu:3g.40gb=1", State="CANCELLED+"),
    ]
    s = G.summarize(rows)
    assert abs(s["by_name"]["rf-render"]["gpu_hours_raw"] - 1.5) < 1e-12
    assert abs(s["by_name"]["rf-render"]["a100_hours"] - (1.0 + 0.5 * 4 / 7)) < 1e-12
    assert abs(s["total"]["a100_hours"] - (1.0 + 0.5 * 4 / 7 + 2 * 3 / 7)) < 1e-12
    assert s["by_name"]["rf-pscan"]["jobs"] == 1 and s["total"]["jobs"] == 3
