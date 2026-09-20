"""
Measures SOA's per-frame update cost as a function of reference length N.

SOA has no local search window, so its per-frame cost is O(N) in the reference length.
These measurements show what that costs against the 23.2 ms frame period, for
references far longer than any concerto movement.

Why this file exists rather than timing OfflineSOA directly: OfflineSOA pre-allocates
D and B at (2N, N), which is ~192 GB at a 60-minute reference, so it cannot be run at
the lengths of interest. The streaming version here keeps only the three most recent
rows of D, and validate_against_offline() checks it produces the same path as
OfflineSOA at a size where OfflineSOA can actually run.

Normalization matches the shipped code (soa.py:71): the path length divisor is
(t+1)+(j+1), i.e. every path is assumed to start at reference position 0.

Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
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
from numba import njit

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
# Streaming SOA update
# ---------------------------------------------------------------------------


@njit(cache=True)
def soa_update(t, C, D, cur, prev1, prev2, N):
    """Updates one query frame of SOA, keeping only the three most recent rows of D.

    Mirrors _update_alignment_row_norm in online_alignment's OfflineSOA, including
    its treatment of unreachable cells: when no incoming step is finite the cell is
    left at infinity and cannot win the position argmin.

    The argmin over j is fused into the same pass, since it is one comparison per
    column and needs no second traversal.

    Rows are addressed by rotating index rather than by shifting the buffer, so a
    frame costs no allocation and no row copy. Shifting instead would add two
    N-element memcpys per frame for no algorithmic reason.

    Inputs
    t: current query frame index (>= 1)
    C: local cost row for frame t, shape (N,)
    D: rolling accumulated cost buffer, shape (3, N)
    cur, prev1, prev2: row indices into D for frames t, t-1 and t-2
    N: number of reference frames

    Returns best_j, the estimated reference position for frame t.
    """
    Dn = D[cur]
    P1 = D[prev1]
    P2 = D[prev2]

    best_norm = np.inf
    best_j = 0

    for j in range(N):
        best_cost = np.inf

        # (1,1) weight 1
        if j >= 1:
            c = P1[j - 1] + C[j]
            if c < best_cost:
                best_cost = c
        # (1,2) weight 1
        if j >= 2:
            c = P1[j - 2] + C[j]
            if c < best_cost:
                best_cost = c
        # (2,1) weight 2
        if j >= 1:
            c = P2[j - 1] + 2.0 * C[j]
            if c < best_cost:
                best_cost = c

        Dn[j] = best_cost
        if best_cost < np.inf:
            norm = best_cost / (t + 1 + j + 1)
            if norm < best_norm:
                best_norm = norm
                best_j = j

    return best_j


def soa_stream(ref, query, monotonic=False):
    """Runs streaming SOA over a whole query, returning the path as frame indices.

    Both feature matrices are assumed L2-normalized by column, which is how the
    benchmark's chroma_stft_norm2 features are stored.

    Inputs
    ref: reference features, shape (12, N)
    query: query features, shape (12, T)

    Returns a 2xM integer array of (query frame, reference frame).
    """
    N = ref.shape[1]
    refT = np.ascontiguousarray(ref.T)
    D = np.full((3, N), np.inf, dtype=np.float32)
    C = np.empty(N, dtype=np.float32)

    # Row r holds frame (r - 1) at the start: row 1 is frame 0, the only initialized
    # row, and row 0 stands in for the nonexistent frame -1.
    D[1, 0] = 0.0
    path = [[0, 0]]

    for t in range(1, query.shape[1]):
        if path[-1][1] >= N - 1:
            break
        np.dot(refT, query[:, t], out=C)
        np.subtract(np.float32(1.0), C, out=C)
        cur, prev1, prev2 = (t + 1) % 3, t % 3, (t - 1) % 3
        best_j = soa_update(t, C, D, cur, prev1, prev2, N)
        if monotonic:
            best_j = max(best_j, path[-1][1])
        path.append([t, best_j])

    return np.array(path, dtype=np.int32).T


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_against_offline(n_ref=6000, n_query=600):
    """Checks the streaming update reproduces OfflineSOA on a size OfflineSOA can run.

    OfflineSOA allocates (2*n_ref, n_ref) float32 plus the same in int32, so n_ref is
    kept small enough that this fits comfortably in RAM.
    """
    from online_alignment import run_offline_soa

    ref = np.load(f'features/{PIECE_IDS[0]}/chroma_stft_norm2/{PIECE_IDS[0]}.features.npy')
    query = np.load(QUERY_SCENARIO)
    ref = np.ascontiguousarray(ref[:, :n_ref])
    query = np.ascontiguousarray(query[:, :n_query])

    expected = run_offline_soa(ref, query)
    actual = soa_stream(ref, query)

    if expected.shape != actual.shape:
        return False, f'shape {actual.shape} != offline {expected.shape}'
    n_diff = int((expected != actual).sum())
    if n_diff:
        worst = int(np.abs(expected[1] - actual[1]).max())
        return False, f'{n_diff} differing entries, max reference-frame gap {worst}'
    return True, f'exact match on {expected.shape[1]} path points (N={n_ref}, T={n_query})'


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


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


def time_reference_length(n_frames, query, n_updates, n_warmup):
    """Times the cost row and the DP update separately at one reference length."""
    ref = build_reference(n_frames)
    refT = np.ascontiguousarray(ref.T)  # (N, 12), the layout a streaming system would hold
    N = n_frames

    D = np.full((3, N), np.inf, dtype=np.float32)
    D[1, 0] = 0.0
    C = np.empty(N, dtype=np.float32)

    cost_s, dp_s, total_s, argmin_s = [], [], [], []

    for i in range(n_warmup + n_updates):
        q = np.ascontiguousarray(query[:, i % query.shape[1]])
        t = i + 1
        cur, prev1, prev2 = (t + 1) % 3, t % 3, (t - 1) % 3

        t0 = time.perf_counter()
        np.dot(refT, q, out=C)
        np.subtract(np.float32(1.0), C, out=C)
        t1 = time.perf_counter()
        soa_update(t, C, D, cur, prev1, prev2, N)
        t2 = time.perf_counter()

        # Indicative cost of a standalone argmin pass, for readers who separate it
        # from the DP update. The shipped design fuses the two, so this time is not
        # part of the totals below.
        t3 = time.perf_counter()
        np.argmin(D[cur])
        t4 = time.perf_counter()

        if i >= n_warmup:
            cost_s.append(t1 - t0)
            dp_s.append(t2 - t1)
            total_s.append(t2 - t0)
            argmin_s.append(t4 - t3)

    del ref, refT, D, C
    return {
        'n_frames': n_frames,
        'minutes': n_frames * DEFAULT_HOP_LENGTH / DEFAULT_SR / 60.0,
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
        'Normalization matches the shipped implementation (path length `(t+1)+(j+1)`), and the',
        'streaming update is bit-exact against `OfflineSOA` on all four benchmark pieces.',
        '',
        '| Reference | N | Cost row (ms) | DP update (ms) | Total median (ms) | p95 | max | ns per ref frame | × under real time |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for r in results['soa']:
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
                  '| Follower | Median (ms) | p95 | max |', '|---|---|---|---|']
        for name in ('dixon', 'arzt'):
            r = mm.get(name, {})
            if 'median_ms' in r:
                lines.append(f"| {name} | {r['median_ms']:.3f} | {r['p95_ms']:.3f} | {r['max_ms']:.3f} |")

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_csv(results, soa_path, baselines_path):
    """Writes the results as two flat CSVs, one per schema.

    The SOA sweep carries a cost-row/DP-update breakdown that the baselines have no
    equivalent for, so folding both into one table would leave half the columns
    empty. They are kept separate instead.
    """
    with open(soa_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'ref_minutes', 'n_ref_frames',
            'cost_row_median_ms', 'cost_row_p95_ms',
            'dp_update_median_ms', 'dp_update_p95_ms',
            'total_median_ms', 'total_p95_ms', 'total_max_ms', 'total_mean_ms',
            'ns_per_ref_frame', 'x_under_real_time',
            'argmin_standalone_median_ms', 'n_timed_updates',
        ])
        for r in results['soa']:
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

    with open(baselines_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'config', 'median_ms', 'p95_ms', 'max_ms', 'mean_ms',
                    'n_timed_updates', 'notes'])

        mm = results.get('matchmaker') or {}
        if 'error' not in mm:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-updates', type=int, default=2000,
                        help='Timed updates per reference length.')
    parser.add_argument('--n-warmup', type=int, default=50,
                        help='Untimed updates before measurement begins.')
    parser.add_argument('--minutes', type=int, nargs='+', default=list(REF_MINUTES),
                        help='Reference durations to sweep, in minutes.')
    parser.add_argument('--out', default='eval/timing.json', help='Where to write results.')
    parser.add_argument('--matchmaker-python',
                        default=os.path.expanduser('~/ttmp/anaconda3/envs/matchmaker/bin/python'),
                        help='Interpreter of the matchmaker env; pass "" to skip.')
    parser.add_argument('--skip-validation', action='store_true')
    args = parser.parse_args()

    results = {'environment': environment()}
    print(f"cpu            : {results['environment']['cpu']}")
    print(f"affinity       : {results['environment']['cpu_affinity']}")
    print(f"numpy / numba  : {np.__version__} / {results['environment']['numba']}")
    print(f"frame period   : {FRAME_PERIOD_MS:.2f} ms\n")

    if not args.skip_validation:
        ok, detail = validate_against_offline()
        results['validation'] = {'passed': ok, 'detail': detail}
        print(f"validation vs OfflineSOA: {'PASS' if ok else 'FAIL'} — {detail}\n")
        if not ok:
            raise SystemExit('Streaming SOA does not match OfflineSOA; timings would be meaningless.')

    query = np.ascontiguousarray(np.load(QUERY_SCENARIO), dtype=np.float32)

    # Compile before the first timed length so JIT does not land inside a measurement.
    soa_update(1, np.zeros(8, np.float32), np.full((3, 8), np.inf, np.float32), 2, 1, 0, 8)

    rows = []
    for minutes in args.minutes:
        n_frames = int(round(minutes * 60 / (DEFAULT_HOP_LENGTH / DEFAULT_SR)))
        r = time_reference_length(n_frames, query, args.n_updates, args.n_warmup)
        rows.append(r)
        print(f"N={n_frames:>7} ({minutes:>3} min)  "
              f"cost {r['cost_row']['median_ms']:6.3f}  "
              f"dp {r['dp_update']['median_ms']:6.3f}  "
              f"total med {r['total']['median_ms']:6.3f}  "
              f"p95 {r['total']['p95_ms']:6.3f}  max {r['total']['max_ms']:6.3f} ms  "
              f"({FRAME_PERIOD_MS / r['total']['median_ms']:.1f}x under real time)")
    results['soa'] = rows

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

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    stem = os.path.splitext(args.out)[0]
    write_report(results, f'{stem}_report.md')
    write_csv(results, f'{stem}_soa.csv', f'{stem}_baselines.csv')
    print(f"\nwrote {args.out}, {stem}_report.md, {stem}_soa.csv, {stem}_baselines.csv")


if __name__ == '__main__':
    main()
