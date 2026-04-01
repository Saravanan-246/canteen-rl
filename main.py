from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from environment import CanteenEnvironment
from agent import QLearningAgent
from config import EPISODES, STEPS_PER_EPISODE
from utils import calculate_efficiency
from model_loader import save_model, load_model


app = FastAPI(title="Canteen RL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    steps: int = 10


ACTION_MAP = {
    0: "No Action",
    1: "Open Counter",
    2: "Speed Service",
    3: "Close Counter"
}


env = CanteenEnvironment()
agent = QLearningAgent(actions=[0, 1, 2, 3])


def train_agent():
    for _ in range(EPISODES):
        state = env.reset()

        for _ in range(STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward, _ = env.step(action)

            agent.learn(state, action, reward, next_state)
            state = next_state

        agent.decay_epsilon()

    agent.epsilon = 0.05
    save_model(agent)


# Try loading existing model, else train
try:
    load_model(agent)
except:
    train_agent()


@app.get("/")
def root():
    return {"message": "Canteen RL API running"}


@app.post("/simulate")
def simulate(data: InputData):
    steps = min(data.steps, 50)

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