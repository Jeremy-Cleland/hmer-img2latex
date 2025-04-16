# This file marks img2latex.analysis as a Python package
"""
Analysis tools for the img2latex project.

This package contains various tools for analyzing model performance,
data distributions, and visualizing results.

Important implementation notes:
-------------------------------

1. All analysis modules should load and use values from the config file
   to ensure consistency with the actual training and inference code:

   - preprocess.py: Uses image dimensions (height, width, channels) from config
   - tokens.py: Uses max_seq_length from config for consistent tokenization
   - errors.py: Can use max_seq_length from config for error analysis

2. When adding new analysis modules, ensure they use the standard pattern:
   - Import config loading helper
   - Take config_path as first parameter with default to "img2latex/configs/config.yaml"
   - Extract and use relevant parameters from the config

This ensures that when training parameters change, analysis modules will automatically
use the updated values, preventing inconsistencies between analysis and actual model behavior.
"""
