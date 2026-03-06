import logging
import os
import librosa as lb
from numba import jit, njit, prange
import numpy as np
import system_utils

logger = logging.getLogger(__name__)


from utils.systems import online_processing_oltw
from hmc_mir.align import dtw
from online_alignment import run_offline_noa, run_offline_oltw
from online_alignment.constants import DEFAULT_DTW_STEPS, DEFAULT_DTW_WEIGHTS, OLTW_STEPS, OLTW_WEIGHTS, NOA_STEPS, NOA_WEIGHTS

DEFAULT_SR = 22050
DEFAULT_HOP_LENGTH = 512

@njit(cache=True)
def compute_cosine_distance(feature_row, reference_features):
    """Compute cosine distance between normalized feature vectors.

    Assumes both feature_row and reference_features are already normalized (unit vectors).
    For normalized vectors, cosine distance = 1 - dot_product.
    """
    costs = np.empty(reference_features.shape[1], dtype=np.float32)

    for j in range(reference_features.shape[1]):
        ref_col = reference_features[:, j]
        dot_product = np.sum(feature_row * ref_col)
        costs[j] = 1.0 - dot_product

    return costs

def verify_cache_dir(indir):
    '''
    Verifies that the specified cache directory has the required files.
    
    Inputs
    indir: The cache directory to verify (features/{piece_id})
    '''
    return


@jit(nopython=True, parallel=True)
def cosine_dist(F1, F2):
    '''
    Calculates the pairwise cosine distance matrix between two features matrices.

    Inputs
    F1: the first feature matrix, shape D x N
    F2: the second feature matrix, shape D x M

    Returns a pairwise cost matrix C of shape N x M, where elements indicate cosine distance.
    '''
    F1 = F1.T
    F2 = F2.T
    C = np.zeros((F1.shape[0], F2.shape[0]))
    for row in prange(F1.shape[0]):
        for col in prange(F2.shape[0]):
            C[row, col] = 1 - np.dot(F1[row], F2[col]) / (np.linalg.norm(F1[row]) * np.linalg.norm(F2[col]) + 1e-9)
    return C

def load_features(scenario_path, p_ref_cache_dir):
    """
    Loads the query and reference features from the given scenario path and p_ref_cache_dir.
    Inputs:
    scenario_path -- str, path to a single scenario file
    p_ref_cache_dir -- str, path to where the features are stored
    Returns:
    F_pquery -- numpy array, query features
    F_pref -- numpy array, reference features
    """
    #load query feautes
    #scenario_path assumes -> scenarios/{benchmark}/{mode}/{s_id}/
    pquery_feat_path = os.path.join(scenario_path, "pquery_stft.npy")
    F_pquery = np.load(pquery_feat_path) 
    #p_ref_cache_dir assumes -> features/{piece_id}/{piece_id}.features.npy
    F_pref = np.load(p_ref_cache_dir) 
    return F_pquery, F_pref

def run_dtw(scenario_path, out_dir, p_ref_cache_dir, hop_length = DEFAULT_HOP_LENGTH, sr = DEFAULT_SR, steps=DEFAULT_DTW_STEPS, weights=DEFAULT_DTW_WEIGHTS):
    '''
    Carries out the 'online' processing for a simple offline subseq DTW system.
    Inputs:
    scenario_path -- str, path to a single scenario file
    out_dir -- str, directory to where the .hyp will be saved
    p_ref_cache_dir -- str, path to where the features are stored
    '''
    logger.info("DTW | scenario=%s  out=%s", scenario_path, out_dir)
    system_utils.verify_scenario_dir(scenario_path)
    # verify_cache_dir(p_ref_cache_dir)
    assert not os.path.exists(out_dir), f'Output directory {out_dir} already exists.'
    os.makedirs(out_dir)

    logger.debug("DTW | loading features from %s and %s", scenario_path, p_ref_cache_dir)
    F_pquery, F_pref = load_features(scenario_path, p_ref_cache_dir)

    logger.debug("DTW | computing cost matrix  F_pquery=%s  F_pref=%s", F_pquery.shape, F_pref.shape)
    C = cosine_dist(F_pquery, F_pref)
    _, _, wp_AB = dtw.dtw(C, steps, weights, subseq=True)

    # convert to seconds
    hop_sec = hop_length / sr
    np.save(f'{out_dir}/hyp.npy', wp_AB*hop_sec)
    logger.info("DTW | saved hyp -> %s/hyp.npy", out_dir)

def run_noa(scenario_path, out_dir, p_ref_cache_dir, ref_start_time, hop_length = DEFAULT_HOP_LENGTH, sr = DEFAULT_SR, steps=NOA_STEPS, weights=NOA_WEIGHTS):
    '''
    Carries out the 'online' processing for a simple offline subseq NOA system.
    Inputs:
    scenario_path -- str, path to a single scenario file
    out_dir -- str, directory to where the .hyp will be saved
    p_ref_cache_dir -- str, path to where the features are stored
    ref_start_time -- float, start time of the reference in seconds
    '''
    logger.info("NOA | scenario=%s  out=%s  ref_start=%.3fs", scenario_path, out_dir, ref_start_time)
    system_utils.verify_scenario_dir(scenario_path)
    verify_cache_dir(p_ref_cache_dir)
    assert not os.path.exists(out_dir), f'Output directory {out_dir} already exists.'
    os.makedirs(out_dir)
    
    logger.debug("NOA | loading features")
    F_pquery, F_pref = load_features(scenario_path, p_ref_cache_dir)
    
    # chop reference features
    hop_sec = hop_length / sr
    F_pref = F_pref[:, int(ref_start_time / hop_sec):]
    logger.debug("NOA | chopped F_pref to shape %s", F_pref.shape)
    
    wp = run_offline_noa(F_pref, F_pquery, steps, weights)
    
    # convert to seconds
    wp_sec = wp * hop_sec
    
    # add offset to wp_sec
    wp_sec[1, :] += ref_start_time
    
    np.save(f'{out_dir}/hyp.npy', wp_sec)
    logger.info("NOA | saved hyp -> %s/hyp.npy", out_dir)
    
def run_noa_monotonic(scenario_path, out_dir, p_ref_cache_dir, ref_start_time, hop_length = DEFAULT_HOP_LENGTH, sr = DEFAULT_SR, steps=NOA_STEPS, weights=NOA_WEIGHTS):
    '''
    Carries out the 'online' processing for a simple offline subseq NOA system with monotonic constraint.
    Inputs:
    scenario_path -- str, path to a single scenario file
    out_dir -- str, directory to where the .hyp will be saved
    p_ref_cache_dir -- str, path to where the features are stored
    ref_start_time -- float, start time of the reference in seconds
    '''
    logger.info("NOA-MONO | scenario=%s  out=%s  ref_start=%.3fs", scenario_path, out_dir, ref_start_time)
    system_utils.verify_scenario_dir(scenario_path)
    verify_cache_dir(p_ref_cache_dir)
    assert not os.path.exists(out_dir), f'Output directory {out_dir} already exists.'
    os.makedirs(out_dir)
    
    logger.debug("NOA-MONO | loading features")
    F_pquery, F_pref = load_features(scenario_path, p_ref_cache_dir)
    
    # chop reference features
    hop_sec = hop_length / sr
    F_pref = F_pref[:, int(ref_start_time / hop_sec):]
    logger.debug("NOA-MONO | chopped F_pref to shape %s", F_pref.shape)
    
    wp = run_offline_noa(F_pref, F_pquery, steps, weights, monotonic=True)
    
    # convert to seconds
    wp_sec = wp * hop_sec
    
    # add offset to wp_sec
    wp_sec[1, :] += ref_start_time
    
    np.save(f'{out_dir}/hyp.npy', wp_sec)
    logger.info("NOA-MONO | saved hyp -> %s/hyp.npy", out_dir)
    
def run_oltw_global(scenario_path, out_dir, p_ref_cache_dir, ref_start_time, hop_length = DEFAULT_HOP_LENGTH, sr = DEFAULT_SR, dtw_steps=OLTW_STEPS, dtw_weights=OLTW_WEIGHTS, window_steps=OLTW_STEPS):
    '''
    Carries out the 'online' processing for OLTW-GLOBAL system.
    Inputs:
    scenario_path -- str, path to a single scenario file
    out_dir -- str, directory to where the .hyp will be saved
    p_ref_cache_dir -- str, path to where the features are stored
    ref_start_time -- float, start time of the reference in seconds
    hop_length -- int, hop length in samples
    sr -- int, sample rate
    dtw_steps -- numpy array, steps for DTW
    dtw_weights -- numpy array, weights for DTW
    window_steps -- numpy array, steps for window
    '''
    logger.info("OLTW-GLOBAL | scenario=%s  out=%s  ref_start=%.3fs", scenario_path, out_dir, ref_start_time)
    system_utils.verify_scenario_dir(scenario_path)
    verify_cache_dir(p_ref_cache_dir)
    assert not os.path.exists(out_dir), f'Output directory {out_dir} already exists.'
    os.makedirs(out_dir)
    
    logger.debug("OLTW-GLOBAL | loading features")
    F_pquery, F_pref = load_features(scenario_path, p_ref_cache_dir)
    
    # chop reference features
    hop_sec = hop_length / sr
    F_pref = F_pref[:, int(ref_start_time / hop_sec):]
    logger.debug("OLTW-GLOBAL | chopped F_pref to shape %s", F_pref.shape)
    
    wp = run_offline_oltw(F_pref, F_pquery, dtw_steps, dtw_weights, window_steps, c=None)
    
    # convert to seconds
    wp_sec = wp * hop_sec
    
    # add offset to wp_sec
    wp_sec[1, :] += ref_start_time
    
    np.save(f'{out_dir}/hyp.npy', wp_sec)
    logger.info("OLTW-GLOBAL | saved hyp -> %s/hyp.npy", out_dir)

def run_oltw(scenario_path, out_dir, hop_length):
    logger.info("OLTW | scenario=%s  out=%s", scenario_path, out_dir)
    online_processing_oltw(scenario_path, out_dir, hop_length, jar_path=None)
    logger.info("OLTW | saved hyp -> %s/hyp.npy", out_dir)
    
    
if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    VALID_BENCHMARKS = ["train"]
    VALID_MODES = ["constant", "continuous", "random"]
    VALID_SYSTEMS = ["DTW", "NOA", "NOA-MONO", "OLTW", "OLTW-GLOBAL"]

    parser = argparse.ArgumentParser(
        description="Run online alignment processing for a given benchmark, mode, and system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        choices=VALID_BENCHMARKS,
        nargs="+",
        default=VALID_BENCHMARKS,
        help="Benchmark split(s) to process.",
    )
    parser.add_argument(
        "--modes",
        choices=VALID_MODES,
        nargs="+",
        default=VALID_MODES,
        help="Tempo-variation mode(s) to process.",
    )
    parser.add_argument(
        "--systems",
        choices=VALID_SYSTEMS,
        nargs="+",
        default=VALID_SYSTEMS,
        help="Alignment system(s) to run.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run every combination of benchmark, mode, and system (overrides --benchmark/--modes/--systems).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging verbosity level (default: WARNING).",
    )
    args = parser.parse_args()

    # Configure logging — auto-generated log file under logs/, matching offline_processing style
    from datetime import datetime
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/online_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=format_str,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger.info("Logging to %s", log_file)

    # --all overrides individual selections
    benchmarks = VALID_BENCHMARKS if args.run_all else args.benchmark
    modes      = VALID_MODES      if args.run_all else args.modes
    systems    = VALID_SYSTEMS    if args.run_all else args.systems

    hop_length = DEFAULT_HOP_LENGTH
    sr = DEFAULT_SR

    total = len(benchmarks) * len(modes) * len(systems)
    logger.info(
        "Starting run: %d benchmark(s) x %d mode(s) x %d system(s) = %d combination(s)",
        len(benchmarks), len(modes), len(systems), total,
    )
    logger.info("benchmarks=%s  modes=%s  systems=%s", benchmarks, modes, systems)

    for benchmark in benchmarks:
        for mode in modes:
            for system in systems:
                scenarios_root   = f"scenarios/{benchmark}/{mode}"
                exp_root         = f"experiments/{benchmark}/{mode}/{system}"
                feature_dir      = "features"
                scenarios_summary = f"{scenarios_root}/scenarios.summary"

                logger.info("=== %s / %s / %s ===", benchmark, mode, system)

                if not os.path.exists(scenarios_summary):
                    logger.warning("Skipping — scenarios summary not found: %s", scenarios_summary)
                    continue

                d = system_utils.get_scenario_info(scenarios_summary, input_type="summary")
                n_scenarios = len(d)
                logger.info("Found %d scenario(s) in %s", n_scenarios, scenarios_root)

                skipped = 0
                for i in tqdm(range(n_scenarios), desc=f"{benchmark}/{mode}/{system}"):
                    scenario_id  = f"s{i+1}"
                    scenario_dir = f"{scenarios_root}/{scenario_id}"
                    out_dir      = f"{exp_root}/{scenario_id}"

                    if os.path.exists(f"{out_dir}/hyp.npy"):
                        logger.debug("Skipping %s — hyp.npy already exists", scenario_id)
                        skipped += 1
                        continue

                    # Derive the reference feature path from the orchestra recording id.
                    o_id = os.path.splitext(os.path.basename(d[scenario_id]["o"]))[0]
                    p_id = "_".join(o_id.split("_")[:-1]) + "_P1"
                    p_ref_cache_dir = f"{feature_dir}/{p_id}/chroma_stft_norm2/{p_id}.features.npy"
                    ref_start_time  = d[scenario_id]["prefStart"]

                    logger.debug(
                        "Processing %s  ref_feat=%s  ref_start=%.3fs",
                        scenario_id, p_ref_cache_dir, ref_start_time,
                    )

                    try:
                        if system == "DTW":
                            run_dtw(scenario_dir, out_dir, p_ref_cache_dir, hop_length=hop_length, sr=sr)
                        elif system == "NOA":
                            run_noa(scenario_dir, out_dir, p_ref_cache_dir, ref_start_time, hop_length=hop_length, sr=sr)
                        elif system == "NOA-MONO":
                            run_noa_monotonic(scenario_dir, out_dir, p_ref_cache_dir, ref_start_time, hop_length=hop_length, sr=sr)
                        elif system == "OLTW":
                            run_oltw(scenario_dir, out_dir, hop_length)
                        elif system == "OLTW-GLOBAL":
                            run_oltw_global(scenario_dir, out_dir, p_ref_cache_dir, ref_start_time, hop_length=hop_length, sr=sr)
                    except Exception:
                        logger.exception("Error processing %s — skipping", scenario_id)

                logger.info(
                    "Finished %s/%s/%s: %d processed, %d skipped",
                    benchmark, mode, system, n_scenarios - skipped, skipped,
                )

    logger.info("All done.")
