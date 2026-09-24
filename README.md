# Piano Concerto Accompaniment

The goal of this project is to generate an orchestral accompaniment recording that is
time-scale modified to match a user's playing without requiring a symbolic
representation of the musical piece.

This repository provides a framework, dataset, and benchmark for the concerto
accompaniment task. The framework supports both the offline and online formulations;
this branch holds the online benchmark, where the accompaniment must follow the soloist
as they play. A system receives piano audio one frame at a time and reports, for each
query frame, the corresponding position in a reference recording.

## Paper

This branch holds the Piano Concerto experiments and the per-frame timing study of
*A Simple Alternative to Online Time Warping*. The alignment algorithms live in the
[OnlineAlignment](https://github.com/HMC-MIR/OnlineAlignment) package (pinned to v0.4.1 in
`environment.yml`), and the paper's Mazurka and Vienna 4x22 experiments are in
[SimRealtimeMazurkaBenchmark](https://github.com/HMC-MIR/SimRealtimeMazurkaBenchmark/tree/vienna4x22).

## Methods evaluated

| System | Description |
| --- | --- |
| **SOA** | Simple Online Alignment. Evaluates the whole reference at every query frame instead of a local search window, so an early mistake does not trap the estimate. |
| **SOA-MONO** | SOA constrained so the reported position cannot move backwards. |
| **OLTW** | On-line time warping as implemented in Simon Dixon's original Java program (MATCH), which computes its own features. |
| **OLTW-GLOBAL** | Our OLTW reimplementation with the search window removed, so the band spans the whole reference. |
| **MM-ARZT** | MatchMaker's Arzt follower. Its position is clipped to a fixed number of reference frames per step, making it monotonic and rate limited. |
| **MM-DIXON** | MatchMaker's frame-level Dixon follower, with its alignment path reduced to one estimate per query time — the reduction MatchMaker's own evaluation applies. |
| **MM-DIXON-RAW** | The same follower reporting its alignment path unreduced (not in the paper). Included to show what a consumer of that array receives; its error numbers are not interpretable as accuracy, because a non-monotone query axis breaks the interpolation the evaluation relies on. |
| **DTW** | Offline subsequence DTW, included as a bound on what an online system could achieve. |

Systems are compared under three tempo-variation modes, which control how the soloist's
timing departs from the reference. Each query is a time-scale-modified copy of a piano
chunk:

| Mode (code) | Paper | How the query is generated |
| --- | --- | --- |
| `constant` | constant | one global factor from {0.8, 0.9, 1.0, 1.1, 1.25} |
| `random` | segmented | the chunk is split into 30 segments, each with a factor drawn uniformly on a log scale from [1/2, 2] |
| `continuous` | continuous | the factor changes by up to ±0.5% per frame as a random walk, clipped to [1/2, 2] |

The mode is called `random` in directory names and code, and "segmented" in the paper.

## Setup

```bash
conda env create -f environment.yml
conda activate PianoConcertoAccompaniment
```

**MatchMaker baselines.** `pymatchmaker` pins `numpy < 2`, so MM-DIXON and MM-ARZT run
out of process in their own environment, which pins `pymatchmaker==0.3.0`:

```bash
conda env create -f environment-matchmaker.yml
export MATCHMAKER_PYTHON=$(conda run -n matchmaker which python)   # optional
```

If `MATCHMAKER_PYTHON` is unset, the `matchmaker` env is looked for next to the active
one.

**OLTW.** The OLTW baseline runs Simon Dixon's original Java implementation from the
MATCH toolkit (Dixon and Widmer, "MATCH: A Music Alignment Tool Chest", ISMIR 2005; main
class `at.ofai.music.match.PerformanceMatcher`). Download `PerformanceMatcher.jar` and
place it at `match/PerformanceMatcher.jar` (the `match/` directory is not committed). It
needs Java 21 or later, which `environment.yml` installs.

## Running the benchmark

1. **Data.** `01_DataPrep.ipynb` downloads and checks the audio and annotations
   (`download_fullmixes.sh` fetches the full mixes). Then run `bash setup_annot_links.sh`,
   which links the piano-only beat annotations to the orchestra ones. Only the four
   movements in `cfg_files/train.list` are used.
2. **Queries, features and systems.** `bash run_train_benchmark.sh` generates the queries
   for all three modes (`generate_data.py`, writing `scenarios/train/{mode}/`), computes
   features (`offline_processing.py`), and runs every system (`online_processing.py --all`,
   writing `experiments/train/{mode}/{system}/`). `online_processing.py --help` shows how
   to run a subset of systems or modes; `--jobs` runs scenarios in parallel.
3. **Evaluation.** `03_Evaluate.ipynb` computes the error rates per system, mode and
   tolerance.

**Timing study.** With the data in place,

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    taskset -c 0 python timing_benchmark.py
```

measures SOA's per-frame cost against reference length, with a flexible and a fixed
start, and the MatchMaker followers for comparison. `documentation/timing_study.md`
explains the method and records the results.

## Layout

| Path | Contents |
| --- | --- |
| `generate_data.py` | Generates the time-scale-modified queries and scenarios for one benchmark and mode. |
| `offline_processing.py` | Computes and caches the features each system needs. |
| `online_processing.py` | Runs a system over the scenarios of one benchmark and mode, writing `hyp.npy` per scenario. |
| `run_train_benchmark.sh` | The whole train benchmark: queries, features and all systems. |
| `eval_tools.py`, `system_utils.py` | Scoring against beat annotations, and scenario helpers. |
| `timing_benchmark.py` | Per-frame latency measurements. |
| `utils/query/` | Query generation for the three tempo-variation modes. |
| `utils/systems/` | Per-system adapters: the OLTW jar wrapper and the MatchMaker workers. |
| `cfg_files/` | Piece lists and the audio data summary. |
| `annot/` | Beat annotations. |
| `01_DataPrep.ipynb` | Downloads and checks the audio and annotations. |
| `03_Evaluate.ipynb` | Error rates per system, mode and tolerance. |
| `04_Analysis.ipynb` | Alignment path inspection. |
| `documentation/timing_study.md` | The per-frame timing study. |
| `environment.yml`, `environment-matchmaker.yml` | The benchmark and MatchMaker environments. |

## Prior work

An earlier study using this framework focused on the offline formulation and introduced
Dense-Sparse DTW, an alignment algorithm robust to additive noise.

TJ Tsai, Kavi Dey, Yigitcan Ozer, and Meinard Mueller. "Dense-Sparse Dynamic Time
Warping for Customizing Piano Concerto Accompaniments" in Proceedings of the IEEE
International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2025,
pp. 1-5.

## License

The code is released under the MIT License; see [LICENSE](LICENSE). The audio recordings
are not part of this repository and keep their own terms.
