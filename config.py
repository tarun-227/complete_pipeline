"""
config.py - All hyperparameters and paths
==========================================
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_GT_DIR = os.path.join(BASE_DIR, "data", "with_gt")
DATA_PRED_DIR = os.path.join(BASE_DIR, "data", "without_gt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Preprocessing
VOXEL_SIZE = 0.5          # Light downsample (set to None for no downsampling)
SOR_NEIGHBORS = 20        # Statistical outlier removal
SOR_STD = 2.0
KNN_NORMALS = 30          # Normal estimation

# Dataset
NUM_POINTS_PER_SAMPLE = 4096    # Points per training sample (local patch)
NEIGHBORHOOD_RADIUS = 5.0       # Radius for extracting local patches
USE_NORMALS = True              # Use normals as input features (XYZ + normals = 6 channels)
INPUT_CHANNELS = 6 if USE_NORMALS else 3  # 3 for XYZ, 6 for XYZ+normals

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 10
PATIENCE = 15              # Early stopping patience

# Class weights (break surface is minority class)
# Will be computed from data, but default fallback
BREAK_WEIGHT = 7.0         # Approximate ratio original:break
ORIGINAL_WEIGHT = 1.0

# Model
DROPOUT = 0.4

# GPU
DEVICE = "cuda:0"
NUM_WORKERS = 4

# Prediction
PRED_BATCH_SIZE = 64
PRED_THRESHOLD = 0.5       # Will be optimized on validation set