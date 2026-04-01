from environment import CanteenEnvironment
from agent import QLearningAgent
from config import EPISODES, STEPS_PER_EPISODE
from utils import calculate_efficiency
from model_loader import save_model


ACTION_MAP = {
    0: "No Action",
    1: "Open Counter",
    2: "Speed Service",
    3: "Close Counter"
}


def train():
    env = CanteenEnvironment()
    agent = QLearningAgent(actions=[0, 1, 2, 3])

    rewards = []
    best = float("-inf")

    print("Training...\n")

    for ep in range(EPISODES):
        state = env.reset()
        total = 0

        for _ in range(STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward, _ = env.step(action)

            agent.learn(state, action, reward, next_state)

            state = next_state
            total += reward

        agent.decay_epsilon()

        rewards.append(total)
        best = max(best, total)

        if (ep + 1) % 10 == 0:
            avg = sum(rewards[-10:]) / 10
            print(
                f"Ep {ep+1:3d} | Avg(10): {avg:7.2f} | "
                f"Best: {best:7.2f} | Eps: {agent.epsilon:.3f}"
            )

    print("\nDone.\n")
    save_model(agent)

    return agent, rewards


def evaluate(agent, runs=3, steps=20):
    print("=== EVALUATION ===\n")

    agent.epsilon = 0.0

    for run in range(runs):
        env = CanteenEnvironment()
        state = env.reset()

        print(f"Run {run + 1}\n")
        total = 0

        for step in range(steps):
            action = agent.choose_action(state)
            next_state, reward, _ = env.step(action)

            info = env.get_info()
            eff = calculate_efficiency(
                info["queue_length"],
                info["counters"]
            )

            print(
                f"Step {step+1:2d} | "
                f"Q:{info['queue_length']:2d} | "
                f"C:{info['counters']} | "
                f"T:{info['time_period']:6s} | "
                f"A:{ACTION_MAP[action]:15s} | "
                f"R:{reward:6.2f} | "
                f"E:{eff:6.2f}"
            )

            total += reward
            state = next_state

        print(f"\nTotal Reward: {total:.2f}\n")


if __name__ == "__main__":
    agent, _ = train()
    evaluate(agent)