# Per-Frame Timing Study

This document records the per-frame latency measurements for SOA and the MatchMaker
OLTW baselines.

SOA has no local search window, so its per-frame cost grows with the reference length
`N`. The measurements below show that it grows linearly, at seven to nine nanoseconds
per reference frame, and that it stays far inside the 23.22 ms frame period even for a
two-hour reference.

Everything here is produced by `timing_benchmark.py` at the repository root. See
[Reproducing](#reproducing) at the end.

## What is measured

`timing_benchmark.py` times the SOA update of the
[online_alignment](https://github.com/HMC-MIR/OnlineAlignment) package (version 0.4.0),
the code that produced the accuracy results, split into two pieces:

* the **cost row**: cosine distance between one 12-dimensional chroma vector and all
  `N` reference frames, and
* the **DP update**: the pass that computes the new row of `D` from the two previous
  rows, the path-length-normalized scores, and the position `argmin`.

The standalone time of the `argmin` alone is reported in the CSV as an indicative figure;
it is already part of the DP update and is not added to the totals. `SOA.feed()` adds
about 0.04 ms of Python bookkeeping per frame (input check, path append) on top of these.

With a fixed start and SOA's steps (1,1), (1,2), (2,1), query frame `t` can only reach
reference frames `t/2` to `2t`. Early in a performance most of the row is still
unreachable, so timing only the first frames understates the cost of a performance in
progress. Each reference length therefore runs `N/2` untimed updates first, where the
reachable part of the row is largest (about three quarters of the reference), and then
2000 timed updates, single-threaded and pinned to one core.

References of each length are built by concatenating the four piano-only benchmark
recordings and tiling the result. Audio content does not affect timing, only length does.

Feature extraction is timed separately. It is part of total latency but not part of the
complexity argument, since every online method including OLTW pays it.

## Correctness gate

`timing_benchmark.py` will not report timings for code it has not first checked.
`validate_against_feed()` runs the timed update loop and the package's `SOA.feed()` on
the same inputs and requires the integer alignment paths to be identical; the script
aborts if they differ. These are therefore timings for the algorithm that produced the
accuracy results, not for a lookalike.

## Results: SOA against reference length

All times in milliseconds: the median of 2000 updates on one pinned core, and then the
median of three complete runs. Frame period is 23.22 ms (hop 512 at 22050 Hz). The
same numbers are in `eval/timing_soa.csv`.

| Reference | N       | Cost row | DP update | Total median | p95   | ns/ref frame | Inside deadline |
|-----------|---------|----------|-----------|--------------|-------|--------------|-----------------|
| 5 min     | 12,920  | 0.046    | 0.070     | 0.115        | 0.144 | 8.93         | 201×            |
| 15 min    | 38,760  | 0.132    | 0.145     | 0.277        | 0.311 | 7.15         | 83.7×           |
| 30 min    | 77,520  | 0.258    | 0.284     | 0.544        | 0.615 | 7.02         | 42.7×           |
| 60 min    | 155,039 | 0.533    | 0.593     | 1.131        | 1.240 | 7.29         | 20.5×           |
| 120 min   | 310,078 | 1.499    | 1.221     | 2.725        | 3.005 | 8.79         | 8.5×            |

The 120-minute row is included to show headroom beyond any realistic concert-length
reference. The three runs agree to within 0.05 ms on every total.

Cost per reference frame stays between 7.0 and 8.9 ns across a 24x range in `N`, so the
cost is linear in the reference length. The rise at 120 minutes is in the cost row: its
reference (12 × 310,078 float32 values, 14.9 MB) no longer fits in the 13.75 MB shared L3
cache of this CPU, so it is streamed from main memory.

Maximum per-frame times are recorded in the CSV but are not reproduced here. They are
noisy on a shared 40-thread machine (the 30-minute row shows a 1.52 ms maximum against a
0.544 ms median), which is a scheduler blip rather than a property of the algorithm.
Median and p95 are the defensible figures.

Feature extraction (`chroma_stft`) costs 0.060 ms per frame. This is librosa batch time
divided by frame count, which is a lower bound on the true streaming per-frame cost
rather than an estimate of it, because librosa has no frame-by-frame API.

## Results: head to head with OLTW

All three systems run on the same core, the same chroma features, the same query stream
and an identical 11.4-minute reference (N = 29429), so the comparison is like for like.
The OLTW baselines use the benchmark's 10-second search window, about 430 frames. Times
are medians of three runs.

| System           | Median | p95    | Positions examined per frame | ns per position |
|------------------|--------|--------|------------------------------|-----------------|
| SOA              | 0.263  | 0.301  | 29,429                       | 8.9             |
| MatchMaker-Arzt  | 0.092  | 0.104  | ~430                         | 214             |
| MatchMaker-Dixon | 9.380  | 10.081 | ~430                         | 21,814          |

Arzt is the efficient OLTW implementation and the meaningful comparison. SOA is 2.9x
slower than Arzt while examining 68x more reference positions, which is roughly 24x more
efficient per position. A dense contiguous sweep over the reference is cheap per
position, while a sliding window spends most of its time on bookkeeping.

At the 60-minute reference the same trade widens: 12x Arzt's cost for 360x the
positions, still 20x inside the deadline.

SOA is also 36x faster than Dixon, but that gap reflects pymatchmaker's implementation
quality rather than OLTW's complexity.

### Dixon warm-up

Dixon's per-frame cost ramps from about 2.8 ms to a plateau of about 9.5 ms over the
first ~500 frames, as its search window grows to full size. A short warm-up reports a
blend of the ramp and the plateau, so `matchmaker_timing_worker.py` defaults to 600
warm-up frames. The SOA sweep warms up for N/2 frames, for the reason given above.

Dixon's cost was separately confirmed to be flat in `N` (measured at N = 6000 through
77520), so it is genuinely O(w). The growth is with elapsed time, not reference length.

## Reproducing

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    taskset -c 0 python timing_benchmark.py
```

Writes four files under `eval/`, which is gitignored, so the tables above are the
tracked copy of these numbers. The tables are the median of three runs of this command;
one run takes about 15 minutes, most of it the warm-up of the 120-minute reference.

* `timing.json` — full results including the environment block and max/mean figures
* `timing_report.md` — the same tables rendered as markdown
* `timing_soa.csv` — the reference-length sweep
* `timing_baselines.csv` — the OLTW followers, SOA on their reference, and feature extraction

Per-frame OLTW timing runs through `utils/systems/matchmaker_timing_worker.py` in the
`matchmaker` conda environment, invoked as a subprocess, because pymatchmaker pins
numpy < 2 and cannot share a process with the benchmark environment. This is the same
arrangement `matchmaker_worker.py` uses.

Environment for the numbers above: Intel Xeon Silver 4210R at 2.40 GHz, single pinned
core, Python 3.11.14, NumPy 2.3.5, numba 0.63.1, online_alignment 0.4.0,
12-dimensional chroma in float32, hop 512 at 22050 Hz.
