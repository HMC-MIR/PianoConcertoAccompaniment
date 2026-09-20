# Piano Concerto Accompaniment

The goal of this project is to generate an orchestral accompaniment recording that is
time-scale modified to match a user's playing without requiring a symbolic
representation of the musical piece.

This repository provides a framework, dataset, and benchmark for the concerto
accompaniment task. The framework supports both the offline and online formulations;
this branch holds the online benchmark, where the accompaniment must follow the soloist
as they play. A system receives piano audio one frame at a time and reports, for each
query frame, the corresponding position in a reference recording.

## Methods evaluated

| System | Description |
| --- | --- |
| **SOA** | Simple Online Alignment. Evaluates the whole reference at every query frame instead of a local search window, so an early mistake does not trap the estimate. |
| **SOA-MONO** | SOA constrained so the reported position cannot move backwards. |
| **OLTW** | On-line time warping as implemented by the MATCH Vamp plugin, which computes its own features. |
| **OLTW-GLOBAL** | Our OLTW reimplementation with the search window removed, so the band spans the whole reference. |
| **MM-ARZT** | MatchMaker's Arzt follower. Its position is clipped to a fixed number of reference frames per step, making it monotonic and rate limited. |
| **MM-DIXON** | MatchMaker's frame-level Dixon follower, with its alignment path reduced to one estimate per query time — the reduction MatchMaker's own evaluation applies. |
| **MM-DIXON-RAW** | The same follower reporting its alignment path unreduced. Included to show what a consumer of that array receives; its error numbers are not interpretable as accuracy, because a non-monotone query axis breaks the interpolation the evaluation relies on. |
| **DTW** | Offline subsequence DTW, included as a bound on what an online system could achieve. |

Systems are compared under three tempo-variation modes — `constant`, `continuous` and
`random` — which control how the soloist's timing departs from the reference. Per-frame
latency is measured separately; see `documentation/timing_study.md`.

## Layout

| Path | Contents |
| --- | --- |
| `online_processing.py` | Runs a system over the scenarios of one benchmark and mode, writing `hyp.npy` per scenario. `--jobs` runs scenarios in parallel. |
| `offline_processing.py` | Offline counterparts of the same systems. |
| `timing_benchmark.py` | Per-frame latency measurements. |
| `utils/systems/` | Per-system adapters, including the MatchMaker workers. |
| `01_DataPrep.ipynb` | Builds scenarios from the audio and annotations. |
| `03_Evaluate.ipynb` | Error rates per system, mode and tolerance. |
| `04_Analysis.ipynb` | Alignment path inspection. |
| `documentation/` | Benchmark stages, per-system notes, and the timing study. |

Alignment results, features, audio and scenarios are gitignored; the notebooks and
`documentation/` hold the numbers that are kept.

## Environment

`environment.yml` covers the benchmark environment. The MatchMaker baselines need a
separate conda environment named `matchmaker` with `pymatchmaker` installed, because it
pins `numpy < 2`; point `MATCHMAKER_PYTHON` at its interpreter if it is not a sibling of
the active environment.

## Prior work

An earlier study using this framework focused on the offline formulation and introduced
Dense-Sparse DTW, an alignment algorithm robust to additive noise.

TJ Tsai, Kavi Dey, Yigitcan Ozer, and Meinard Mueller. "Dense-Sparse Dynamic Time
Warping for Customizing Piano Concerto Accompaniments" in Proceedings of the IEEE
International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2025,
pp. 1-5.
