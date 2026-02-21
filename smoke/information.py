TRIAL = 8   # MODIFY THIS BEFORE EVERY RUN
MODEL_SAVE_PATH = f'saved_models/dehazer_trial_{TRIAL}.keras'
HISTORY_SAVE_PATH = f'saved_histories/dehazer_history_trial{TRIAL}.pkl'
NUM_EPOCHS = 50
BATCH_SIZE = 16
INPUT_SHAPE = (256, 256, 3)