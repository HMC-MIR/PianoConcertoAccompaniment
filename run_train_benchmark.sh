# Generate queries
python generate_data.py --benchmark train --mode constant --tsm_factor 0.8 0.9 1.0 1.1 1.25
python generate_data.py --benchmark train --mode random --max_tsm_factor 2.0
python generate_data.py --benchmark train --mode continuous --max_alpha_change 1.005

# Run offline processing
python offline_processing.py

# Run online processing
python online_processing.py --all