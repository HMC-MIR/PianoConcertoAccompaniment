#!/usr/bin/env python
"""
Query and Scenario Generation Script

This standalone script generates queries and alignment scenarios for the 
Piano Concerto Accompaniment benchmark. It replaces the notebook-based workflow
and supports three generation modes: constant, random, and continuous.

Usage:
    python generate_data.py --benchmark train --mode constant
    python generate_data.py --benchmark test --mode random --max_tsm_factor 2.5
"""

import argparse
import os
import re
import shutil
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from utils.query import QueryGenerator, get_query_timestamps, get_audio_files
from utils.query.scenario import (
    generateScenariosConstant,
    generateScenariosRandom,
    generateScenariosContinuous
)


class DataGenerator:
    """Main class for handling query and scenario generation."""
    
    # Default configuration values
    DEFAULT_TSM_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.25]
    DEFAULT_MAX_TSM_FACTOR = 2
    DEFAULT_MAX_ALPHA_CHANGE = 1.002
    DEFAULT_NUM_QUERIES_TRAIN = 5
    DEFAULT_NUM_QUERIES_TEST = 10
    
    # Directory structure
    DIRS = {
        'audio': 'audio',
        'annot': 'annot',
        'queries': 'queries',
        'scenarios': 'scenarios',
        'cfg_files': 'cfg_files',
        'logs': 'logs'
    }
    
    # Config files
    CONFIG_FILES = {
        'train': 'cfg_files/train.list',
        'test': 'cfg_files/test.list',
        'audio_summary': 'cfg_files/AudioDataSummary.csv',
        'query_measures': 'annot/query.measures'
    }
    
    def __init__(self, benchmark, mode, args):
        """Initialize the data generator."""
        self.benchmark = benchmark
        self.mode = mode
        self.args = args
        self.logger = None
        self.setup_logging()
        self.setup_parameters()
        self.validate_directories()
        
    def setup_logging(self):
        """Set up logging to console and file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"logs/generate_{self.benchmark}_{self.mode}_{timestamp}.log"
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_format)
        
        # File handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_filename
        
    def setup_parameters(self):
        """Set up generation parameters based on mode and defaults."""
        if self.mode == 'constant':
            self.tsm_factors = self.DEFAULT_TSM_FACTORS
            if self.args.tsm_factor:
                self.logger.info(f"Using custom TSM factors: {self.args.tsm_factor}")
                # Parse the list if provided
                if isinstance(self.args.tsm_factor, list):
                    self.tsm_factors = self.args.tsm_factor
                    
        elif self.mode == 'random':
            self.max_tsm_factor = self.args.max_tsm_factor or self.DEFAULT_MAX_TSM_FACTOR
            self.logger.info(f"Using max_tsm_factor: {self.max_tsm_factor}")
            
        elif self.mode == 'continuous':
            self.max_alpha_change = self.args.max_alpha_change or self.DEFAULT_MAX_ALPHA_CHANGE
            self.logger.info(f"Using max_alpha_change: {self.max_alpha_change}")
    
    def validate_directories(self):
        """Validate that required directories exist."""
        self.logger.info("Validating directory structure...")
        required_dirs = [
            self.DIRS['audio'],
            self.DIRS['annot'],
            self.DIRS['cfg_files']
        ]
        
        for dir_name in required_dirs:
            if not os.path.isdir(dir_name):
                self.logger.error(f"Required directory not found: {dir_name}")
                raise FileNotFoundError(f"Directory {dir_name} does not exist")
        
        # Create queries and scenarios directories if they don't exist
        os.makedirs(self.DIRS['queries'], exist_ok=True)
        os.makedirs(self.DIRS['scenarios'], exist_ok=True)
        
        self.logger.info("Directory structure validated successfully")
    
    def validate_config_files(self):
        """Validate that required config files exist."""
        self.logger.info("Validating configuration files...")
        
        config_file = self.CONFIG_FILES[self.benchmark]
        if not os.path.isfile(config_file):
            self.logger.error(f"Config file not found: {config_file}")
            raise FileNotFoundError(f"Config file {config_file} does not exist")
        
        if not os.path.isfile(self.CONFIG_FILES['audio_summary']):
            self.logger.error(f"Audio summary file not found: {self.CONFIG_FILES['audio_summary']}")
            raise FileNotFoundError(f"Audio summary file {self.CONFIG_FILES['audio_summary']} does not exist")
        
        if not os.path.isfile(self.CONFIG_FILES['query_measures']):
            self.logger.error(f"Query measures file not found: {self.CONFIG_FILES['query_measures']}")
            raise FileNotFoundError(f"Query measures file {self.CONFIG_FILES['query_measures']} does not exist")
        
        self.logger.info("Configuration files validated successfully")
    
    def get_piece_ids(self):
        """Parse and return list of piece IDs from config file."""
        config_file = self.CONFIG_FILES[self.benchmark]
        piece_ids = []
        
        with open(config_file, 'r') as f:
            for line in f:
                piece_id = line.strip()
                if piece_id:  # Skip empty lines
                    piece_ids.append(piece_id)
        
        self.logger.info(f"Found {len(piece_ids)} piece(s) to process")
        return piece_ids
    
    def get_num_queries(self):
        """Return the number of queries based on benchmark."""
        if self.benchmark == 'train':
            return self.DEFAULT_NUM_QUERIES_TRAIN
        else:  # test
            return self.DEFAULT_NUM_QUERIES_TEST
    
    def get_query_seeds(self):
        """Return the range of seeds based on benchmark."""
        num_queries = self.get_num_queries()
        if self.benchmark == 'train':
            return list(range(0, num_queries))
        else:  # test
            return list(range(100, 100 + num_queries))
    
    def count_piano_recordings(self):
        """Count the number of piano-only recordings available."""
        count = 0
        audio_dir = self.DIRS['audio']
        
        for filename in os.listdir(audio_dir):
            if '_P' in filename and filename.endswith('.wav'):
                count += 1
        
        self.logger.info(f"Found {count} piano-only recording(s)")
        return count
    
    def generate_queries(self):
        """Generate queries for the specified mode."""
        self.logger.info(f"Starting query generation for {self.mode} mode...")
        
        # Initialize QueryGenerator
        qgen = QueryGenerator(
            audio_summary_file=self.CONFIG_FILES['audio_summary'],
            query_measures_file=self.CONFIG_FILES['query_measures'],
            audio_root=self.DIRS['audio'],
            annot_root=self.DIRS['annot']
        )
        
        # Create benchmark-specific query directory
        queries_benchmark_dir = os.path.join(self.DIRS['queries'], self.benchmark)
        os.makedirs(queries_benchmark_dir, exist_ok=True)
        
        try:
            if self.mode == 'constant':
                self.logger.info(f"Generating queries with constant TSM factors: {self.tsm_factors}")
                qgen.generateQueriesConstant(queries_benchmark_dir, self.tsm_factors)
                
            elif self.mode == 'random':
                num_queries = self.get_num_queries()
                self.logger.info(f"Generating {num_queries} random queries per piece with max_tsm_factor={self.max_tsm_factor}, and num_chunks={30}")
                qgen.generateQueriesRandom(queries_benchmark_dir, self.max_tsm_factor, num_queries)
                
            elif self.mode == 'continuous':
                num_queries = self.get_num_queries()
                self.logger.info(f"Generating {num_queries} continuous queries per piece with max_alpha_change={self.max_alpha_change}")
                qgen.generateQueriesContinuous(queries_benchmark_dir, self.max_alpha_change, num_queries)
            
            self.logger.info("Query generation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during query generation: {e}")
            raise
    
    def validate_queries(self):
        """Validate generated queries."""
        self.logger.info("Validating generated queries...")
        
        piece_ids = self.get_piece_ids()
        queries_benchmark_dir = os.path.join(self.DIRS['queries'], self.benchmark)
        
        validation_passed = True
        total_queries = 0
        
        for piece_id in piece_ids:
            piece_queries_dir = os.path.join(queries_benchmark_dir, f"{piece_id}_P1")
            
            if not os.path.isdir(piece_queries_dir):
                self.logger.warning(f"No queries directory found for piece: {piece_id}")
                validation_passed = False
                continue
            
            # Count queries for this piece
            audio_files = [f for f in os.listdir(piece_queries_dir) if f.endswith('.wav')]
            beats_files = [f for f in os.listdir(piece_queries_dir) if f.endswith('.beats')]
            
            if len(audio_files) == 0:
                self.logger.warning(f"No audio files found for piece: {piece_id}")
                validation_passed = False
            else:
                self.logger.info(f"  {piece_id}: {len(audio_files)} audio files")
                total_queries += len(audio_files)
            
            if len(beats_files) == 0:
                self.logger.warning(f"No beats files found for piece: {piece_id}")
                validation_passed = False
            else:
                self.logger.info(f"  {piece_id}: {len(beats_files)} beats files")
        
        self.logger.info(f"Total queries generated: {total_queries}")
        
        if validation_passed:
            self.logger.info("Query validation passed")
        else:
            self.logger.warning("Query validation completed with warnings")
        
        return validation_passed
    
    def generate_scenarios(self):
        """Generate alignment scenarios."""
        self.logger.info(f"Starting scenario generation for {self.mode} mode...")
        
        piece_ids = self.get_piece_ids()
        audio_summary_file = self.CONFIG_FILES['audio_summary']
        
        # Create benchmark and mode subdirectories
        scenarios_benchmark_dir = os.path.join(self.DIRS['scenarios'], self.benchmark)
        scenarios_mode_dir = os.path.join(scenarios_benchmark_dir, self.mode)
        os.makedirs(scenarios_mode_dir, exist_ok=True)
        
        # Get the queries directory for this benchmark
        queries_benchmark_dir = os.path.join(self.DIRS['queries'], self.benchmark)
        
        # Track scenarios
        scenario_count = 0
        log_info = []
        
        try:
            for piece_id in piece_ids:
                self.logger.info(f"Processing scenarios for piece: {piece_id}")
                
                # Find full mix file (PO1)
                fullmix_files = get_audio_files(
                    audio_summary_file,
                    f'^{piece_id}_PO1\\.\\S+$'
                )
                
                if not fullmix_files:
                    self.logger.warning(f"No full mix file found for piece: {piece_id}")
                    continue
                
                # Convert to wav if needed
                fullmix_file = fullmix_files[0]
                fullmix_file = re.sub(r'\.mp3$', '.wav', fullmix_file)
                
                if self.mode == 'constant':
                    res = generateScenariosConstant(
                        cnt=scenario_count,
                        piece_id=piece_id,
                        fullmix_file=fullmix_file,
                        tsm_factors=self.tsm_factors,
                        outdir=scenarios_mode_dir,
                        QUERIES_ROOT=queries_benchmark_dir,
                        ANNOT_ROOT=self.DIRS['annot'],
                        AUDIO_ROOT=self.DIRS['audio'],
                        QUERY_MEASURES_FILE=self.CONFIG_FILES['query_measures']
                    )
                    scenario_count = res['cnt']
                    log_info.extend(res.get('logInfo', []))
                    
                elif self.mode == 'random':
                    num_queries = self.get_num_queries()
                    res = generateScenariosRandom(
                        cnt=scenario_count,
                        num_queries=num_queries,
                        piece_id=piece_id,
                        fullmix_file=fullmix_file,
                        max_tsm_factors=[self.max_tsm_factor],
                        outdir=scenarios_mode_dir,
                        QUERIES_ROOT=queries_benchmark_dir,
                        ANNOT_ROOT=self.DIRS['annot'],
                        AUDIO_ROOT=self.DIRS['audio'],
                        QUERY_MEASURES_FILE=self.CONFIG_FILES['query_measures']
                    )
                    scenario_count = res['cnt']
                    log_info.extend(res.get('logInfo', []))
                    
                elif self.mode == 'continuous':
                    num_queries = self.get_num_queries()
                    res = generateScenariosContinuous(
                        cnt=scenario_count,
                        num_queries=num_queries,
                        piece_id=piece_id,
                        fullmix_file=fullmix_file,
                        max_alpha_changes=[self.max_alpha_change],
                        outdir=scenarios_mode_dir,
                        QUERIES_ROOT=queries_benchmark_dir,
                        ANNOT_ROOT=self.DIRS['annot'],
                        AUDIO_ROOT=self.DIRS['audio'],
                        QUERY_MEASURES_FILE=self.CONFIG_FILES['query_measures']
                    )
                    scenario_count = res['cnt']
                    log_info.extend(res.get('logInfo', []))
            
            # write summary file if we collected any entries
            if log_info:
                summary_path = os.path.join(scenarios_mode_dir, 'scenarios.summary')
                with open(summary_path, 'w') as sf:
                    for ln in log_info:
                        sf.write(ln)
                self.logger.info(f"Summary written to {summary_path}")
            self.logger.info(f"Scenario generation completed. Total scenarios: {scenario_count}")
            
        except Exception as e:
            self.logger.error(f"Error during scenario generation: {e}")
            raise
    
    def run(self):
        """Execute the full data generation pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("Piano Concerto Accompaniment - Data Generator")
        self.logger.info("=" * 60)
        self.logger.info(f"Benchmark: {self.benchmark}")
        self.logger.info(f"Mode: {self.mode}")
        
        try:
            # Validate configuration
            self.validate_config_files()
            
            # Count available recordings
            self.count_piano_recordings()
            
            # Stage 1: Generate queries
            self.generate_queries()
            
            # Validate queries
            self.validate_queries()
            
            # Stage 2: Generate scenarios
            self.generate_scenarios()
            
            self.logger.info("=" * 60)
            self.logger.info("Data generation completed successfully!")
            self.logger.info(f"Log file: {self.log_file}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error("=" * 60)
            self.logger.error(f"Data generation failed: {e}")
            self.logger.error("=" * 60)
            raise


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate queries and scenarios for the Piano Concerto Accompaniment benchmark'
    )
    
    # Required arguments
    parser.add_argument(
        '--benchmark',
        required=True,
        choices=['train', 'test'],
        help='Benchmark to generate data for'
    )
    
    parser.add_argument(
        '--mode',
        required=True,
        choices=['constant', 'random', 'continuous'],
        help='Query generation mode'
    )
    
    # Optional arguments for constant mode
    parser.add_argument(
        '--tsm_factor',
        type=float,
        nargs='+',
        help='TSM factors for constant mode (space-separated, e.g., 0.8 0.9 1.0 1.1 1.25)'
    )
    
    # Optional arguments for random mode
    parser.add_argument(
        '--max_tsm_factor',
        type=float,
        help='Maximum TSM factor for random mode (default: 2)'
    )
    
    # Optional arguments for continuous mode
    parser.add_argument(
        '--max_alpha_change',
        type=float,
        help='Maximum alpha change for continuous mode (default: 1.005)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create and run the data generator
    generator = DataGenerator(args.benchmark, args.mode, args)
    generator.run()


if __name__ == '__main__':
    main()
