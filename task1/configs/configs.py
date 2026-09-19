# task1/configss/configs.py

SEED = 6304

# Paths
DATA_ROOT = "./data"
OUTPUT_DIR = "./task1/results"
SPLITS_FILE = "./task1/data/stl10_splits.json"

# Training Settings
BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Dataset Partitioning
TRAIN_VAL_SPLIT_RATIO = 0.20
TEST_SUBSET_SIZE = 500

# Laptop Testing / Quick Debugging
SMOKE_TEST = False  # Set to True to use small subsets and limit batch passes
SMOKE_TEST_SAMPLES = 50
MAX_TRAIN_BATCHES = 2  # Used when SMOKE_TEST = True
MAX_VAL_BATCHES = 2  # Used when SMOKE_TEST = True