import random
from config import MAX_COUNTERS, IDEAL_QUEUE, BALANCE_WEIGHT


class CanteenEnvironment:
    def __init__(self):
        self.max_counters = MAX_COUNTERS

        self.time_map = {"low": 0, "medium": 1, "peak": 2}

        self.arrival_ranges = {
            "low": (1, 3),
            "medium": (3, 6),
            "peak": (5, 8)
        }

        self.service_range = (2, 4)

        self.reset()

    def reset(self):
        self.queue_length = random.randint(5, 20)
        self.counters = 1
        self.time_period = random.choice(["low", "medium", "peak"])
        return self.get_state()

    def get_state(self):
        return (
            min(self.queue_length // 5, 4),
            self.counters,
            self.time_map[self.time_period]
        )

    def step(self, action):
        reward = 0
        prev_queue = self.queue_length

        # ----- ACTION PHASE -----
        if action == 1:  # Open Counter
            if self.counters < self.max_counters:
                self.counters += 1
                reward += 4 if self.queue_length > 10 else -2
            else:
                reward -= 4

        elif action == 2:  # Speed Service
            served = random.randint(4, 8)
            self.queue_length = max(0, self.queue_length - served)

            # prevent overuse of speed
            reward += 2
            reward -= 1

        elif action == 3:  # Close Counter
            if self.counters > 1:
                self.counters -= 1
                reward += 4 if self.queue_length < 5 else -4
            else:
                reward -= 5

        # ----- ARRIVAL PHASE -----
        low, high = self.arrival_ranges[self.time_period]
        self.queue_length += random.randint(low, high)

        # ----- SERVICE PHASE -----
        served = self.counters * random.randint(*self.service_range)
        self.queue_length = max(0, self.queue_length - served)

        # Cap queue (stability)
        self.queue_length = min(self.queue_length, 50)

        # ----- CORE REWARD -----
        if self.queue_length <= 5:
            reward += 10
        elif self.queue_length <= 12:
            reward += 3
        else:
            reward -= 10

        # Trend reward
        reward += 2 if self.queue_length < prev_queue else -2

        # ----- BALANCE CONTROL -----
        reward -= abs(self.queue_length - IDEAL_QUEUE) * BALANCE_WEIGHT

        # ----- SMART LOGIC -----

        # Encourage opening in heavy load
        if self.queue_length > 12 and self.counters < 2:
            reward -= 3

        # Encourage closing when idle
        if self.queue_length < 5 and self.counters > 1:
            reward -= 2

        # Prevent waste of max counters
        if self.counters == self.max_counters and self.queue_length < 6:
            reward -= 3

        # Peak-hour intelligence  
        if self.time_period == "peak" and self.queue_length < 8:
            reward += 3

        # Mild penalty for doing nothing when crowded
        if action == 0 and self.queue_length > 10:
            reward -= 3

        # Small randomness
        reward += random.uniform(-0.2, 0.2)

        # Time changes
        if random.random() < 0.1:
            self.time_period = random.choice(["low", "medium", "peak"])

        return self.get_state(), round(reward, 2), False

    def get_info(self):
        return {
            "queue_length": self.queue_length,
            "counters": self.counters,
            "time_period": self.time_period
        }