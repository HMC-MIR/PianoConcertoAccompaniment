# Per-Frame Timing Study

This document records the per-frame latency measurements for SOA (called NOA in the
code) and the MatchMaker OLTW baselines.

SOA has no local search window, so its per-frame cost grows with the reference length
`N`. The measurements below show that it grows linearly, at roughly ten nanoseconds per
reference frame, and that it stays far inside the 23.22 ms frame period even for a
two-hour reference.

Everything here is produced by `timing_benchmark.py` at the repository root. See
[Reproducing](#reproducing) at the end.

## What is measured

The DP update is timed in isolation, split into two pieces:

* the **cost row** — cosine distance between one 12-dimensional chroma vector and all
  `N` reference frames, and
* the **DP update** — the pass over `D` that computes the new row.

The position `argmin` is fused into the DP pass, because it is one comparison per
column and needs no second traversal of the row. It is reported separately in the JSON
output only as an indicative standalone figure, and is not part of the totals.

Each reference length runs 2000 timed updates after 50 warm-up frames, single-threaded
and pinned to one core, so the numbers are conservative and reproducible. References of
each length are built by concatenating the four piano-only benchmark recordings and
tiling the result. Audio content does not affect timing, only length does.

Feature extraction is timed separately. It is part of total latency but not part of the
complexity argument, since every online method including OLTW pays it.

## Correctness gate

`timing_benchmark.py` will not report timings for an algorithm it has not first checked.
`validate_against_offline()` runs the streaming implementation and the shipped
`OfflineNOA` on the same inputs and requires the integer alignment paths to be
identical; the script aborts if they differ.

The streaming implementation is bit-exact against `OfflineNOA` on all four benchmark
pieces (N up to 10000, query length up to 2000). These are therefore timings for the
algorithm that produced the accuracy results, not for a lookalike.

## Why a separate implementation exists

`OfflineNOA.align` pre-allocates `D` and `B` at `(2N, N)`. At a 60-minute reference
(N = 155039) that is about 192 GB for `D` alone, and as much again for `B`. It cannot be
run at the lengths these measurements concern.

The three-row streaming form of the algorithm is not implemented anywhere else in this
repository, so `timing_benchmark.soa_update` was written for this study. It keeps only
the three most recent rows of `D` and addresses them by rotating index, so a frame costs
no allocation and no row copy.

## Results: SOA against reference length

All times in milliseconds, median of 2000 updates on one pinned core. Frame period is
23.22 ms (hop 512 at 22050 Hz).

| Reference | N       | Cost row | DP update | Total median | p95   | ns/ref frame | Inside deadline |
|-----------|---------|----------|-----------|--------------|-------|--------------|-----------------|
| 5 min     | 12,920  | 0.069    | 0.059     | 0.131        | 0.160 | 10.15        | 177×            |
| 15 min    | 38,760  | 0.197    | 0.150     | 0.349        | 0.432 | 9.00         | 66.5×           |
| 30 min    | 77,520  | 0.400    | 0.281     | 0.687        | 0.852 | 8.86         | 33.8×           |
| 60 min    | 155,039 | 0.868    | 0.573     | 1.440        | 1.708 | 9.29         | 16.1×           |
| 120 min   | 310,078 | 2.109    | 0.962     | 3.069        | 3.424 | 9.90         | 7.6×            |

The 120-minute row is included to show headroom beyond any realistic concert-length
reference.

Cost per reference frame is flat at 8.9–10.2 ns across a 24x range in `N`, with no
trend, so the linear scaling claim can be stated without qualification.

Maximum per-frame times are recorded in the JSON output but are not reproduced here.
They are noisy on a shared 40-thread machine — the 5-minute row shows a 1.53 ms maximum
against a 0.131 ms median, which is a scheduler blip rather than a property of the
algorithm. Median and p95 are the defensible figures.

Feature extraction (`chroma_stft`) costs 0.067 ms per frame. This is librosa batch time
divided by frame count, which is a lower bound on the true streaming per-frame cost
rather than an estimate of it, because librosa has no frame-by-frame API.

## Results: head to head with OLTW

All three systems run on the same core, the same chroma features, the same query stream
and an identical 11.4-minute reference (N = 29429), so the comparison is like for like.
The OLTW baselines use the benchmark's 10-second search window, about 430 frames.

| System           | Median | p95   | Positions examined per frame | ns per position |
|------------------|--------|-------|------------------------------|-----------------|
| SOA              | 0.322  | 0.366 | 29,429                       | 10.9            |
| MatchMaker-Arzt  | 0.096  | 0.110 | ~430                         | 223             |
| MatchMaker-Dixon | 9.389  | 9.982 | ~430                         | 21,835          |

Arzt is the efficient OLTW implementation and the meaningful comparison. SOA is 3.4x
slower than Arzt while examining 68x more reference positions, which is roughly 20x more
efficient per position. A dense contiguous sweep is what a vectorized inner loop does
well, while a sliding window spends most of its time on bookkeeping.

At the 60-minute reference the same trade widens: 15x Arzt's cost for 360x the
positions, still 16x inside the deadline.

SOA is also 29x faster than Dixon, but that gap reflects pymatchmaker's implementation
quality rather than OLTW's complexity.

### Dixon warm-up

Dixon's per-frame cost ramps from about 2.8 ms to a plateau of about 9.5 ms over the
first ~500 frames, as its search window grows to full size. A short warm-up reports a
blend of the ramp and the plateau, so `matchmaker_timing_worker.py` defaults to 600
warm-up frames where the SOA sweep uses 50.

Dixon's cost was separately confirmed to be flat in `N` (measured at N = 6000 through
77520), so it is genuinely O(w). The growth is with elapsed time, not reference length.

## Reproducing

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 0 python timing_benchmark.py
```

Writes four files under `eval/`, which is gitignored, so the tables above are the
tracked copy of these numbers:

* `timing.json` — full results including the environment block and max/mean figures
* `timing_report.md` — the same tables rendered as markdown
* `timing_soa.csv` — the reference-length sweep
* `timing_baselines.csv` — the OLTW followers and feature extraction

Per-frame OLTW timing runs through `utils/systems/matchmaker_timing_worker.py` in the
`matchmaker` conda environment, invoked as a subprocess, because pymatchmaker pins
numpy < 2 and cannot share a process with the benchmark environment. This is the same
arrangement `matchmaker_worker.py` uses.

Environment for the numbers above: Intel Xeon Silver 4210R at 2.40 GHz, single pinned
core, Python 3.12.2, NumPy 1.26.4, numba 0.60.0, 12-dimensional chroma in float32, hop
512 at 22050 Hz.
