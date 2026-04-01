from environment import CanteenEnvironment
from utils import calculate_efficiency


ACTION_MAP = {
    0: "No Action",
    1: "Open Counter",
    2: "Speed Service",
    3: "Close Counter"
}


def run_simulation(agent, steps=10):
    steps = min(steps, 50)

    env = CanteenEnvironment()
    state = env.reset()

    results = []
    total_reward = 0

    for step in range(steps):
        action = agent.choose_action(state)
        next_state, reward, _ = env.step(action)

        info = env.get_info()
        efficiency = calculate_efficiency(
            info["queue_length"],
            info["counters"]
        )

        results.append({
            "step": step + 1,
            "queue": info["queue_length"],
            "counters": info["counters"],
            "time": info["time_period"],
            "decision": ACTION_MAP[action],
            "reward": round(reward, 2),
            "efficiency": efficiency
        })

        total_reward += reward
        state = next_state

    avg_eff = (
        sum(r["efficiency"] for r in results) / len(results)
        if results else 0
    )

    if total_reward > 80:
        status = "Excellent"
    elif total_reward > 40:
        status = "Good"
    else:
        status = "Needs Improvement"

    return {
        "results": results,
        "summary": {
            "total_reward": round(total_reward, 2),
            "avg_efficiency": round(avg_eff, 2),
            "status": status
        }
    }