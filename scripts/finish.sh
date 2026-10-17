#!/bin/bash
# The finishing sequence once the cluster renders are done (run the stages one at a time; each is restartable).
#   stage 1: pull + merge          stage 2: metrics + figures        stage 3: pack
#   stage 4: replay plan -> cluster -> apply (featured starts, video pair)
#   stage 5: hero images           stage 6: video                    stage 7: README
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
KFS="kf03 kf04 kf05 kf06 kf07"
stage=${1:?usage: scripts/finish.sh STAGE (1..7)}

case "$stage" in
1) for n in hero $KFS; do alice/sync.sh pull $n; $PY -m render.tilerun merge --spec data/cluster/$n/spec.json --out data/cluster/$n; done ;;
2) $PY scripts/compute_metrics.py --hero data/cluster/hero --keyframes $(for k in $KFS; do echo data/cluster/$k; done) --out results/metrics.json
   $PY scripts/scale_dependence.py --hero data/cluster/hero --chase data/cluster/chase2
   $PY scripts/summarize_scans.py
   $PY scripts/gpu_hours.py
   $PY scripts/cost_estimate.py
   $PY scripts/replay_agreement.py --chase data/cluster/chase2 --run data/cluster/hero --n 160 --out results/replay_agreement.json ;;
3) rm -rf site/data
   $PY scripts/build_pack.py --hero data/cluster/hero --keyframes $(for k in $KFS; do echo data/cluster/$k; done) --out site/data
   $PY scripts/site_text.py --pack site/data ;;
4) $PY scripts/plan_replay.py --hero data/cluster/hero --keyframes $(for k in $KFS; do echo data/cluster/$k; done)
   rsync -a results/replay/replay_in.json alice2:reasoning-fractals/data/
   alice/sync.sh push
   echo "now: ssh alice2 'cd ~/reasoning-fractals && sbatch alice/replay.slurm'; when it finished: alice/sync.sh pull replay; scripts/finish.sh 4b" ;;
4b) $PY scripts/apply_replay.py --pack site/data --keyframes $(for k in $KFS; do echo data/cluster/$k; done) ;;
5) $PY scripts/make_hero.py --run data/cluster/hero --way "${WAY:-series}" --out media --crop "${CROP:-0.2,0.0,0.8}" ;;
6) $PY scripts/make_video.py --pack site/data --way "${WAY:-series}" --out media/zoom-1080x1350.mp4 --pair results/video/pair.json \
     --captions scripts/video_captions.json --square ;;
7) $PY scripts/build_readme.py ;;
*) echo "unknown stage $stage"; exit 1 ;;
esac
