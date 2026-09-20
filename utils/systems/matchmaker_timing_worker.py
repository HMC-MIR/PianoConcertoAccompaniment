"""
Times MatchMaker's OLTW followers per query frame, for the SOA complexity comparison.

Runs in the `matchmaker` conda env for the same reason matchmaker_worker.py does:
pymatchmaker pins numpy<2 and cannot share a process with the benchmark env. Results
come back as JSON on stdout.

Unlike matchmaker_worker.py this does not produce an alignment for evaluation. It
feeds real query frames to a follower one at a time and records how long each
__call__ takes, so that SOA's per-frame cost can be compared against OLTW's.

Nothing in this directory may be named matchmaker.py: it would shadow the installed
package, since the worker's own directory is first on sys.path.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from matchmaker.dp import OnlineTimeWarpingArztFrame, OnlineTimeWarpingDixonFrame

FOLLOWERS = {'dixon': OnlineTimeWarpingDixonFrame, 'arzt': OnlineTimeWarpingArztFrame}

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_SR = 22050
DEFAULT_HOP_LENGTH = 512
FRAME_RATE = DEFAULT_SR / DEFAULT_HOP_LENGTH

# Reference recording used for the OLTW timings. OLTW's per-frame cost depends on its
# window size rather than on N, so one representative length is enough; N is reported
# so the ratio against SOA is checkable.
PIECE_ID = 'rach2_mov1_P1'
QUERY = 'scenarios/train/constant/s1/pquery_stft.npy'

WINDOW_SIZE = 10.0       # seconds, the benchmark's setting
DISTANCE_METRIC = 'cosine'
STEP_SIZE = 3            # arzt only, as in online_processing.run_matchmaker


def summarize(samples_s):
    a = np.asarray(samples_s) * 1e3
    return {
        'median_ms': float(np.median(a)),
        'p95_ms': float(np.percentile(a, 95)),
        'max_ms': float(a.max()),
        'mean_ms': float(a.mean()),
        'n': int(a.size),
    }


def time_follower(method, ref_feat, query_feat, n_updates, n_warmup):
    """Feeds query frames to one follower one at a time, timing each __call__.

    Constructor arguments mirror matchmaker_worker.align() exactly, so the timings
    describe the same configuration the accuracy baselines were run under.
    """
    ref = np.ascontiguousarray(ref_feat.T, dtype=np.float32)
    query = np.ascontiguousarray(query_feat.T, dtype=np.float32)
    ref_secs = np.arange(ref.shape[0]) / FRAME_RATE

    kwargs = {
        'reference_features': ref,
        'score_positions': ref_secs,
        'ref_frame_to_beat': ref_secs,
        'frame_rate': FRAME_RATE,
        'window_size': WINDOW_SIZE,
        'distance_func': DISTANCE_METRIC if method == 'dixon' else DISTANCE_METRIC.capitalize(),
    }
    if method == 'arzt':
        kwargs['step_size'] = STEP_SIZE

    follower = FOLLOWERS[method](**kwargs)

    times = []
    n_frames = 0
    for i in range(n_warmup + n_updates):
        t = i % query.shape[0]
        t0 = time.perf_counter()
        follower(query[t], i / FRAME_RATE)
        dt = time.perf_counter() - t0
        if i >= n_warmup:
            times.append(dt)
        n_frames += 1
        if not follower.is_still_following():
            # Reaching the end of the reference stops the follower; report how far
            # it got rather than timing a follower that is no longer working.
            break

    out = summarize(times) if times else {'error': 'follower stopped before warmup ended'}
    out['frames_fed'] = n_frames
    out['stopped_early'] = not follower.is_still_following()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-updates', type=int, default=2000)
    # Dixon's per-frame cost ramps from ~2.8 ms to a ~9.5 ms plateau over the first
    # ~500 frames, as its search window grows to full size. A short warmup would
    # report a blend of the ramp and the plateau rather than the steady-state cost.
    parser.add_argument('--n-warmup', type=int, default=600)
    args = parser.parse_args()

    ref = np.load(f'{REPO}/features/{PIECE_ID}/chroma_stft_norm2/{PIECE_ID}.features.npy')
    query = np.load(f'{REPO}/{QUERY}')

    results = {
        'reference_frames': int(ref.shape[1]),
        'reference_minutes': float(ref.shape[1] * DEFAULT_HOP_LENGTH / DEFAULT_SR / 60.0),
        'window_size_seconds': WINDOW_SIZE,
        'numpy': np.__version__,
    }

    for method in ('dixon', 'arzt'):
        try:
            results[method] = time_follower(method, ref, query,
                                            args.n_updates, args.n_warmup)
        except Exception as e:
            results[method] = {'error': repr(e)}

    json.dump(results, sys.stdout, indent=2)


if __name__ == '__main__':
    main()
