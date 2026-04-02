import random
import json
import ast
from collections import defaultdict


class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.n_actions = len(actions)

        self.q_table = defaultdict(lambda: [0.0] * self.n_actions)
        self.visit_counts = defaultdict(lambda: [0] * self.n_actions)

        self.alpha = 0.1
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.last_action = None

    # ε-greedy + tie-break + avoid repeat
    def choose_action(self, state):
        q_values = self.q_table[state]

        if random.random() < self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            noisy = [q + random.uniform(0, 1e-6) for q in q_values]
            max_q = max(noisy)
            best = [i for i, q in enumerate(noisy) if q == max_q]

            if self.last_action in best and len(best) > 1:
                best.remove(self.last_action)

            action = random.choice(best)

        self.last_action = action
        return action

    # adaptive LR + clipped update + stable TD target
    def learn(self, state, action, reward, next_state):
        self.visit_counts[state][action] += 1

        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state]) if next_state in self.q_table else 0

        lr = self.alpha / (1 + 0.01 * self.visit_counts[state][action])
        target = reward + self.gamma * max_next_q

        new_q = current_q + lr * (target - current_q)

        # stability clip
        self.q_table[state][action] = max(-500, min(500, new_q))

    # decay exploration
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # best action per state
    def get_policy(self):
        return {
            state: q.index(max(q))
            for state, q in self.q_table.items()
        }

    # save model
    def save(self, path="q_table.json"):
        data = {str(k): v for k, v in self.q_table.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # load model safely
    def load(self, path="q_table.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                parsed = {ast.literal_eval(k): v for k, v in data.items()}
                self.q_table = defaultdict(
                    lambda: [0.0] * self.n_actions, parsed
                )
        except FileNotFoundError:
            print("No saved model found.")

    # debug info
    def get_state_info(self, state):
        return {
            "q_values": [round(q, 3) for q in self.q_table[state]],
            "visits": self.visit_counts[state],
            "epsilon": round(self.epsilon, 4),
        }