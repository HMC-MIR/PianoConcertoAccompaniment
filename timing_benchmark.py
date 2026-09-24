"""
Measures SOA's per-frame update cost as a function of reference length N.

SOA has no local search window, so its per-frame cost is O(N) in the reference length.
These measurements show what that costs against the 23.2 ms frame period, for
references far longer than any concerto movement.

What is timed is the online_alignment package's SOA (the code that produced the
accuracy results): its cost row and its vectorized DP update, called exactly as
SOA.feed() calls them. validate_against_feed() checks that the timed loop reproduces
SOA.feed()'s path before any timing is reported.

Both start modes are timed. With a fixed start and SOA's steps (1,1), (1,2), (2,1),
query frame t can only reach reference frames t/2 to 2t. Early frames therefore leave
most of the row unreachable, which a performance in progress does not, so fixed-start
timing starts at t = N/2 by default, where the reachable part of the row is largest
(three quarters of the reference). With a flexible start almost the whole row is
reachable from the first frames (all but the first t/2 reference frames), and timing
starts after 1000 frames.

Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
        taskset -c 0 python timing_benchmark.py
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import time

import numpy as np

import online_alignment as oa
from online_alignment.alignment.algs.soa import (
    soa_scores_fixed,
    soa_scores_flexible,
    soa_update_fixed,
    soa_update_flexible,
)

DEFAULT_SR = 22050
DEFAULT_HOP_LENGTH = 512
FRAME_PERIOD_MS = 1000.0 * DEFAULT_HOP_LENGTH / DEFAULT_SR  # 23.2 ms

# Reference durations to sweep, in minutes. 120 is there to show headroom beyond any
# realistic concert-length reference.
REF_MINUTES = (5, 15, 30, 60, 120)

# The four piano-only reference recordings of the concerto benchmark.
PIECE_IDS = ('bach5_mov1_P1', 'beeth1_mov1_P1', 'mozart21_mov1_P1', 'rach2_mov1_P1')

QUERY_SCENARIO = 'scenarios/train/constant/s1/pquery_stft.npy'


# ---------------------------------------------------------------------------
# The package's SOA update, step by step
# ---------------------------------------------------------------------------


class TimedSOA:
    """The package's SOA update, split into its two timed halves.

    Uses the cost function and buffers of an online_alignment SOA object and calls
    the same kernels, in the same order, as SOA.feed(). Only the input check and the
    path bookkeeping of feed() are left out (about 0.04 ms per frame), and so is the
    stop at the end of the reference, so that timing can continue past it.
    """

    def __init__(self, ref, flexible=False):
        self.soa = oa.SOA(ref, flexible_start=flexible)  # default steps and weights
        self.flexible = flexible
        self.t = -1

    def cost_row(self, q):
        return self.soa._costs(q)

    def dp_update(self, costs):
        """Computes the next row and returns the estimated reference position."""
        self.t += 1
        t, soa = self.t, self.soa
        if t == 0:
            # the first frame: a fixed start begins at (0, 0), a flexible one anywhere
            if not self.flexible:
                return 0
            soa._D[0] = costs
            soa._S[0] = np.arange(soa.reference_length)
            return int(np.argmin(costs))
        cur, r1, r2 = t % 3, (t - 1) % 3, (t - 2) % 3
        if self.flexible:
            soa_update_flexible(t, costs, soa._D, soa._S, cur, r1, r2, *soa._w)
            soa_scores_flexible(t, soa._D[cur], soa._S[cur], soa._scores)
        else:
            soa_update_fixed(costs, soa._D, cur, r1, r2, *soa._w)
            soa_scores_fixed(t, soa._D[cur], soa._scores)
        return int(np.argmin(soa._scores))


def validate_against_feed(n_ref=6000, n_query=600):
    """Checks the timed update loop reproduces SOA.feed()'s path exactly, in both start modes."""
    ref = np.load(f'features/{PIECE_IDS[0]}/chroma_stft_norm2/{PIECE_IDS[0]}.features.npy')[:, :n_ref]
    query = np.load(QUERY_SCENARIO)[:, :n_query]

    details = []
    for flexible in (False, True):
        expected = oa.run_offline_soa(ref, query, flexible_start=flexible)
        timed = TimedSOA(ref, flexible)
        path = []
        for t in range(query.shape[1]):
            if path and path[-1][1] >= ref.shape[1] - 1:
                break
            path.append([t, timed.dp_update(timed.cost_row(query[:, t]))])
        actual = np.array(path, dtype=np.int64).T
        mode = 'flexible' if flexible else 'fixed'
        if expected.shape != actual.shape or not np.array_equal(expected, actual):
            return False, f'{mode} start: path differs from SOA.feed'
        details.append(f'{mode} start: exact match on {expected.shape[1]} path points')
    return True, '; '.join(details) + f' (N={n_ref}, T={n_query})'


def build_reference(n_frames):
    """Builds a reference of exactly n_frames by concatenating and tiling real chroma.

    Content does not affect timing, but using real features keeps the data
    distribution (and so any denormal-float behaviour) realistic.
    """
    parts = [np.load(f'features/{p}/chroma_stft_norm2/{p}.features.npy') for p in PIECE_IDS]
    pool = np.concatenate(parts, axis=1)
    reps = int(np.ceil(n_frames / pool.shape[1]))
    ref = np.tile(pool, (1, reps))[:, :n_frames]
    return np.ascontiguousarray(ref, dtype=np.float32)


def summarize(samples_s):
    """Converts a list of per-frame durations in seconds to a millisecond summary."""
    a = np.asarray(samples_s) * 1e3
    return {
        'median_ms': float(np.median(a)),
        'p95_ms': float(np.percentile(a, 95)),
        'max_ms': float(a.max()),
        'mean_ms': float(a.mean()),
        'n': int(a.size),
    }


def time_reference_length(n_frames, query, n_updates, n_warmup=None, flexible=False):
    """Times the cost row and the DP update separately at one reference length.

    n_warmup defaults to N/2 untimed updates with a fixed start, where the reachable
    part of the row is largest, and to 1000 with a flexible start, which reaches all
    but the first t/2 reference frames.
    """
    if n_warmup is None:
        n_warmup = 1000 if flexible else n_frames // 2
    timed = TimedSOA(build_reference(n_frames), flexible)
    timed.dp_update(timed.cost_row(np.ascontiguousarray(query[:, 0])))  # the first frame
    scores = timed.soa._scores

    cost_s, dp_s, total_s, argmin_s = [], [], [], []
    for i in range(n_warmup + n_updates):
        q = np.ascontiguousarray(query[:, (i + 1) % query.shape[1]])

        t0 = time.perf_counter()
        costs = timed.cost_row(q)
        t1 = time.perf_counter()
        timed.dp_update(costs)
        t2 = time.perf_counter()

        # Indicative cost of the position argmin alone. It is part of the DP update
        # above, so this time is not added to the totals.
        t3 = time.perf_counter()
        np.argmin(scores)
        t4 = time.perf_counter()

        if i >= n_warmup:
            cost_s.append(t1 - t0)
            dp_s.append(t2 - t1)
            total_s.append(t2 - t0)
            argmin_s.append(t4 - t3)

    reachable = float(np.isfinite(timed.soa._D[timed.t % 3]).mean())
    del timed
    return {
        'n_frames': n_frames,
        'minutes': n_frames * DEFAULT_HOP_LENGTH / DEFAULT_SR / 60.0,
        'start': 'flexible' if flexible else 'fixed',
        'n_warmup': n_warmup,
        'reachable_fraction': reachable,
        'cost_row': summarize(cost_s),
        'dp_update': summarize(dp_s),
        'argmin_standalone': summarize(argmin_s),
        'total': summarize(total_s),
    }


def time_feature_extraction(n_seconds=120.0):
    """Times chroma_stft extraction, reported as batch time divided by frame count.

    librosa has no frame-by-frame streaming API, so this is a lower bound on the true
    per-frame cost of a streaming implementation, not an estimate of it. It is
    reported because it is part of total latency, not because it bears on the
    complexity argument: every online method, OLTW included, pays it.
    """
    import librosa as lb

    for name in ('beeth1_mov1_P1.wav', 'bach5_mov1_P1.wav'):
        path = f'audio/{name}'
        if os.path.exists(path):
            break
    else:
        return None

    y, sr = lb.load(path, sr=DEFAULT_SR, duration=n_seconds)
    lb.feature.chroma_stft(y=y[:sr], sr=sr, hop_length=DEFAULT_HOP_LENGTH)  # warm up

    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        F = lb.feature.chroma_stft(y=y, sr=sr, hop_length=DEFAULT_HOP_LENGTH)
        runs.append(time.perf_counter() - t0)

    n_frames = F.shape[1]
    per_frame_ms = np.array(runs) / n_frames * 1e3
    return {
        'audio': path,
        'seconds_of_audio': float(len(y) / sr),
        'n_frames': int(n_frames),
        'per_frame_ms_median': float(np.median(per_frame_ms)),
        'per_frame_ms_min': float(per_frame_ms.min()),
        'note': 'batch time / frame count; a lower bound on streaming per-frame cost',
    }


def environment():
    """Records the machine and library versions the timings were taken on."""
    import numba
    cpu = ''
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    return {
        'cpu': cpu,
        'cpu_affinity': affinity,
        'platform': platform.platform(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'numba': numba.__version__,
        'online_alignment': oa.__version__,
        'dtype': 'float32',
        'thread_env': {k: os.environ.get(k) for k in
                       ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        'frame_period_ms': FRAME_PERIOD_MS,
    }


# ---------------------------------------------------------------------------
# MatchMaker OLTW comparison
# ---------------------------------------------------------------------------


def time_matchmaker(env_python, n_updates, n_warmup):
    """Times one MatchMaker OLTW follower per frame, via the matchmaker conda env.

    pymatchmaker pins numpy<2 so it cannot share a process with this one; the worker
    runs as a subprocess and returns its summary as JSON on stdout, the same
    arrangement utils/systems/matchmaker_worker.py uses.
    """
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'utils', 'systems', 'matchmaker_timing_worker.py')
    # The worker sets its own warmup default; Dixon needs a much longer one than SOA.
    cmd = [env_python, worker, '--n-updates', str(n_updates)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                               'OPENBLAS_NUM_THREADS': '1'})
    if proc.returncode != 0:
        return {'error': proc.stderr.strip()[-2000:]}
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


# Keys of the results for each start mode: the sweep, and SOA on the MatchMaker reference
MODES = {
    'fixed': ('soa', 'soa_at_matchmaker_reference'),
    'flexible': ('soa_flexible', 'soa_flexible_at_matchmaker_reference'),
}


def write_report(results, path):
    """Renders the JSON results as markdown tables."""
    env = results['environment']
    lines = [
        '# SOA per-frame timing',
        '',
        f"Frame period: **{FRAME_PERIOD_MS:.2f} ms** (hop {DEFAULT_HOP_LENGTH} @ {DEFAULT_SR} Hz).",
        '',
        f"Machine: {env['cpu']}, pinned to core(s) {env['cpu_affinity']}. "
        f"Python {env['python']}, NumPy {env['numpy']}, numba {env['numba']}, {env['dtype']} features.",
        '',
        f"Times the online_alignment {env['online_alignment']} SOA update, checked against "
        '`SOA.feed()`. Fixed start is timed from query frame N/2 on, where the reachable part',
        'of the row is largest; flexible start reaches all but the first t/2 reference frames',
        'and is timed after 1000 frames.',
    ]
    for mode, (key, _) in MODES.items():
        if key not in results:
            continue
        lines += [
            '',
            f'## {mode.capitalize()} start',
            '',
            '| Reference | N | Cost row (ms) | DP update (ms) | Total median (ms) | p95 | max | ns per ref frame | × under real time |',
            '|---|---|---|---|---|---|---|---|---|',
        ]
        for r in results[key]:
            t = r['total']
            lines.append(
                f"| {r['minutes']:.0f} min | {r['n_frames']:,} | {r['cost_row']['median_ms']:.3f} | "
                f"{r['dp_update']['median_ms']:.3f} | **{t['median_ms']:.3f}** | {t['p95_ms']:.3f} | "
                f"{t['max_ms']:.3f} | {t['median_ms'] * 1e6 / r['n_frames']:.2f} | "
                f"{FRAME_PERIOD_MS / t['median_ms']:.1f}× |"
            )

    feat = results.get('feature_extraction')
    if feat:
        lines += ['', f"Feature extraction (chroma_stft): **{feat['per_frame_ms_median']:.3f} ms/frame** "
                      f"— {feat['note']}. Shared by every online method, SOA and OLTW alike."]

    mm = results.get('matchmaker')
    if mm and 'error' not in mm:
        lines += ['', '## MatchMaker OLTW, same setup',
                  '',
                  f"Reference {mm['reference_frames']:,} frames ({mm['reference_minutes']:.1f} min), "
                  f"window {mm['window_size_seconds']:.0f} s "
                  f"(~{int(mm['window_size_seconds'] * DEFAULT_SR / DEFAULT_HOP_LENGTH):,} frames).",
                  '',
                  '| System | Median (ms) | p95 | max |', '|---|---|---|---|']
        for mode, (_, mm_key) in MODES.items():
            soa_mm = results.get(mm_key)
            if soa_mm:
                t = soa_mm['total']
                lines.append(f"| SOA, {mode} start | {t['median_ms']:.3f} | {t['p95_ms']:.3f} | {t['max_ms']:.3f} |")
        for name in ('dixon', 'arzt'):
            r = mm.get(name, {})
            if 'median_ms' in r:
                lines.append(f"| MatchMaker-{name} | {r['median_ms']:.3f} | {r['p95_ms']:.3f} | {r['max_ms']:.3f} |")

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_soa_csv(rows, path):
    """Writes one start mode's reference-length sweep as a flat CSV."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'ref_minutes', 'n_ref_frames',
            'cost_row_median_ms', 'cost_row_p95_ms',
            'dp_update_median_ms', 'dp_update_p95_ms',
            'total_median_ms', 'total_p95_ms', 'total_max_ms', 'total_mean_ms',
            'ns_per_ref_frame', 'x_under_real_time',
            'argmin_standalone_median_ms', 'n_timed_updates',
        ])
        for r in rows:
            t = r['total']
            w.writerow([
                f"{r['minutes']:.0f}", r['n_frames'],
                f"{r['cost_row']['median_ms']:.4f}", f"{r['cost_row']['p95_ms']:.4f}",
                f"{r['dp_update']['median_ms']:.4f}", f"{r['dp_update']['p95_ms']:.4f}",
                f"{t['median_ms']:.4f}", f"{t['p95_ms']:.4f}",
                f"{t['max_ms']:.4f}", f"{t['mean_ms']:.4f}",
                f"{t['median_ms'] * 1e6 / r['n_frames']:.2f}",
                f"{FRAME_PERIOD_MS / t['median_ms']:.1f}",
                f"{r['argmin_standalone']['median_ms']:.4f}", t['n'],
            ])


def write_baselines_csv(results, path):
    """Writes the OLTW followers, SOA on their reference, and feature extraction.

    The SOA sweeps carry a cost-row/DP-update breakdown that the baselines have no
    equivalent for, so folding both into one table would leave half the columns
    empty. They are kept separate instead.
    """
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'config', 'median_ms', 'p95_ms', 'max_ms', 'mean_ms',
                    'n_timed_updates', 'notes'])

        mm = results.get('matchmaker') or {}
        for mode, (_, mm_key) in MODES.items():
            soa_mm = results.get(mm_key)
            if soa_mm:
                t = soa_mm['total']
                w.writerow([
                    f'soa-{mode}', f"full reference, ref {soa_mm['n_frames']} frames",
                    f"{t['median_ms']:.4f}", f"{t['p95_ms']:.4f}",
                    f"{t['max_ms']:.4f}", f"{t['mean_ms']:.4f}", t['n'],
                    f'{mode} start; cost row + DP update, same reference as the MatchMaker rows',
                ])
        if mm and 'error' not in mm:
            window_frames = int(mm.get('window_size_seconds', 0) * DEFAULT_SR / DEFAULT_HOP_LENGTH)
            for name in ('dixon', 'arzt'):
                r = mm.get(name, {})
                if 'median_ms' not in r:
                    continue
                note = ('per-frame cost ramps to a plateau over the first ~500 frames'
                        if name == 'dixon' else '')
                w.writerow([
                    f'matchmaker-{name}',
                    f"window {mm['window_size_seconds']:.0f}s (~{window_frames} frames), "
                    f"ref {mm['reference_frames']} frames",
                    f"{r['median_ms']:.4f}", f"{r['p95_ms']:.4f}",
                    f"{r['max_ms']:.4f}", f"{r['mean_ms']:.4f}", r['n'], note,
                ])

        feat = results.get('feature_extraction')
        if feat:
            w.writerow([
                'chroma_stft', f"{feat['seconds_of_audio']:.0f}s audio, "
                               f"{feat['n_frames']} frames",
                f"{feat['per_frame_ms_median']:.4f}", '', '', '', feat['n_frames'],
                feat['note'] + '; shared by every online method',
            ])


def print_row(r):
    print(f"N={r['n_frames']:>7} ({r['minutes']:>3.0f} min, {r['start']} start, from frame "
          f"{r['n_warmup']}, {100 * r['reachable_fraction']:.0f}% reachable)  "
          f"cost {r['cost_row']['median_ms']:6.3f}  "
          f"dp {r['dp_update']['median_ms']:6.3f}  "
          f"total med {r['total']['median_ms']:6.3f}  "
          f"p95 {r['total']['p95_ms']:6.3f}  max {r['total']['max_ms']:6.3f} ms  "
          f"({FRAME_PERIOD_MS / r['total']['median_ms']:.1f}x under real time)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-updates', type=int, default=2000,
                        help='Timed updates per reference length.')
    parser.add_argument('--n-warmup', type=int, default=None,
                        help='Untimed updates before measurement begins (default: N/2 with a '
                             'fixed start, where the reachable part of the row is largest; '
                             '1000 with a flexible start).')
    parser.add_argument('--minutes', type=int, nargs='+', default=list(REF_MINUTES),
                        help='Reference durations to sweep, in minutes.')
    parser.add_argument('--start', choices=('fixed', 'flexible', 'both'), default='both',
                        help='Which SOA start modes to time.')
    parser.add_argument('--out', default='eval/timing.json', help='Where to write results.')
    parser.add_argument('--matchmaker-python',
                        default=os.path.expanduser('~/ttmp/anaconda3/envs/matchmaker/bin/python'),
                        help='Interpreter of the matchmaker env; pass "" to skip.')
    parser.add_argument('--skip-validation', action='store_true')
    args = parser.parse_args()
    modes = ('fixed', 'flexible') if args.start == 'both' else (args.start,)

    results = {'environment': environment()}
    print(f"cpu            : {results['environment']['cpu']}")
    print(f"affinity       : {results['environment']['cpu_affinity']}")
    print(f"numpy / numba  : {np.__version__} / {results['environment']['numba']}")
    print(f"online_alignment: {oa.__version__}")
    print(f"frame period   : {FRAME_PERIOD_MS:.2f} ms\n")

    if not args.skip_validation:
        ok, detail = validate_against_feed()
        results['validation'] = {'passed': ok, 'detail': detail}
        print(f"validation vs SOA.feed: {'PASS' if ok else 'FAIL'} — {detail}\n")
        if not ok:
            raise SystemExit('The timed update does not match SOA.feed; timings would be meaningless.')

    query = np.ascontiguousarray(np.load(QUERY_SCENARIO), dtype=np.float32)

    # Compile before the first timed length so JIT does not land inside a measurement.
    for flexible in (False, True):
        warm = TimedSOA(np.ones((12, 8), np.float32), flexible)
        for _ in range(3):
            warm.dp_update(warm.cost_row(np.ones(12, np.float32)))

    for mode in modes:
        rows = []
        for minutes in args.minutes:
            n_frames = int(round(minutes * 60 / (DEFAULT_HOP_LENGTH / DEFAULT_SR)))
            r = time_reference_length(n_frames, query, args.n_updates, args.n_warmup,
                                      flexible=mode == 'flexible')
            rows.append(r)
            print_row(r)
        results[MODES[mode][0]] = rows
        print()

    feat = time_feature_extraction()
    results['feature_extraction'] = feat
    if feat:
        print(f"chroma_stft    : {feat['per_frame_ms_median']:.4f} ms/frame "
              f"(batch proxy, {feat['n_frames']} frames)")

    if args.matchmaker_python:
        mm = time_matchmaker(args.matchmaker_python, args.n_updates, args.n_warmup)
        results['matchmaker'] = mm
        if 'error' in mm:
            print(f"matchmaker     : FAILED — {mm['error'][:300]}")
        else:
            for name, r in mm.items():
                if isinstance(r, dict) and 'median_ms' in r:
                    print(f"matchmaker {name:<8}: median {r['median_ms']:.3f} ms  "
                          f"p95 {r['p95_ms']:.3f}  max {r['max_ms']:.3f}")
            # SOA on the same reference length, for a like-for-like comparison
            for mode in modes:
                soa_mm = time_reference_length(mm['reference_frames'], query, args.n_updates,
                                               args.n_warmup, flexible=mode == 'flexible')
                results[MODES[mode][1]] = soa_mm
                print(f"soa {mode:<8} (same ref): median {soa_mm['total']['median_ms']:.3f} ms  "
                      f"p95 {soa_mm['total']['p95_ms']:.3f}  max {soa_mm['total']['max_ms']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    stem = os.path.splitext(args.out)[0]
    written = [args.out, f'{stem}_report.md', f'{stem}_baselines.csv']
    write_report(results, written[1])
    write_baselines_csv(results, written[2])
    for mode in modes:
        path = f'{stem}_soa.csv' if mode == 'fixed' else f'{stem}_soa_flexible.csv'
        write_soa_csv(results[MODES[mode][0]], path)
        written.append(path)
    print('\nwrote ' + ', '.join(written))


if __name__ == '__main__':
    main()
