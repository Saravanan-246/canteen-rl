import gradio as gr
import matplotlib.pyplot as plt

from agent import QLearningAgent
from service import run_simulation
from config import EPISODES, STEPS_PER_EPISODE
from model_loader import load_model, save_model
from environment import CanteenEnvironment


# Init
env = CanteenEnvironment()
agent = QLearningAgent(actions=[0, 1, 2, 3])


# Train
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


# Load or train
try:
    load_model(agent)
except:
    train_agent()


# Graph
def generate_plot(results):
    steps = [r["step"] for r in results]
    queues = [r["queue"] for r in results]

    plt.figure()
    plt.plot(steps, queues)
    plt.title("Queue Length Over Time")
    plt.xlabel("Step")
    plt.ylabel("Queue")
    plt.grid()

    return plt


# Simulation
def simulate(steps):
    result = run_simulation(agent, steps)

    rows = [
        [
            r["step"],
            r["queue"],
            r["counters"],
            r["time"],
            r["decision"],
            r["reward"],
            r["efficiency"],
        ]
        for r in result["results"]
    ]

    plot = generate_plot(result["results"])

    summary = (
        f"Total Reward: {result['summary']['total_reward']}\n"
        f"Average Efficiency: {result['summary']['avg_efficiency']}\n"
        f"Status: {result['summary']['status']}"
    )

    return rows, summary, plot


# UI
with gr.Blocks() as demo:
    gr.Markdown("# AI Canteen Optimization System")
    gr.Markdown("Reinforcement Learning-based smart queue and counter control")

    with gr.Row():
        steps_input = gr.Slider(5, 30, value=10, step=1, label="Simulation Steps")

    with gr.Row():
        run_btn = gr.Button("Run Simulation")
        retrain_btn = gr.Button("Retrain Model")

    table = gr.Dataframe(
        headers=["Step", "Queue", "Counters", "Time", "Decision", "Reward", "Efficiency"],
        datatype=["number", "number", "number", "str", "str", "number", "number"]
    )

    summary_box = gr.Textbox(label="Summary")

    plot_output = gr.Plot(label="Queue Trend")

    run_btn.click(
        fn=simulate,
        inputs=steps_input,
        outputs=[table, summary_box, plot_output]
    )

    retrain_btn.click(
        fn=train_agent,
        inputs=[],
        outputs=[]
    )


demo.launch()