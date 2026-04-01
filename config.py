# Training
EPISODES = 300
STEPS_PER_EPISODE = 60

# Environment
MAX_COUNTERS = 3

# RL Hyperparameters
ALPHA = 0.12          # learning rate (slightly faster learning)
GAMMA = 0.95          # discount factor

# Exploration
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

# Reward tuning  
IDEAL_QUEUE = 6
OVERFLOW_PENALTY = -10
BALANCE_WEIGHT = 0.5