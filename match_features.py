import subprocess
import pickle
import tempfile
import os
import numpy as np

def extract_match_features(wav_path, output_path=None):
    """
    Extracts features from an audio file using the MATCH Java tool.

    Args:
        wav_path (str): Path to the input WAV file.
        output_path (str, optional): Path to save the extracted features.
    Returns:
        list: Extracted features as strings.
    """
    with tempfile.NamedTemporaryFile(mode='r+', delete=False) as tmp_file:
        temp_output = tmp_file.name

    # Use absolute path for JAR file to ensure it works from any directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jar_path = os.path.join(script_dir, "match", "PerformanceMatcher.jar")
    
    # Fallback to relative path if absolute doesn't exist
    if not os.path.exists(jar_path):
        jar_path = "match/PerformanceMatcher.jar"
    
    cmd = [
        "xvfb-run",
        "java",
        "-jar",
        jar_path,
        "-b", "-q", "-D",
        wav_path,
        wav_path  # Dummy second argument
    ]

    # Run the command and redirect stdout to the temp file
    # Also capture stderr for error reporting
    try:
        with open(temp_output, 'w') as out_f:
            result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, check=True, text=True)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else "No error message available"
        raise RuntimeError(f"Failed to extract match features: {error_msg}\nCommand: {' '.join(cmd)}") from e

    # Read and process the output as floats
    with open(temp_output, 'r') as f:
        lines = f.readlines()
        
        # Process lines: skip lines containing "alignment" and process feature lines
        # Each feature line is a string of comma-separated floats
        # Note that the last character of each line is typically a comma and a newline character
        features = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and lines containing "alignment" (case-insensitive)
            if not line or "alignment" in line.lower():
                continue
            
            # Process feature lines (comma-separated floats)
            # Remove trailing comma if present
            if line.endswith(','):
                line = line[:-1]
            
            # Convert to list of floats
            try:
                feature_values = list(map(float, line.split(",")))
                features.append(feature_values)
            except ValueError:
                # Skip lines that can't be parsed as floats
                continue
        
    # Reshape the features into a 2D array and transpose
    # Each row is a time frame, each column is a feature dimension
    features = np.array(features).T
    
    if features.shape[0] != 84:
        features = np.vstack((features, np.zeros((84 - features.shape[0], features.shape[1]))))

    # Optionally save to pickle
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(features, f)

    # Clean up temp file
    os.remove(temp_output)

    return features