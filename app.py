import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent import QLearningAgent
from service import run_simulation
from model_loader import load_model
from environment import CanteenEnvironment


# INIT
env = CanteenEnvironment()
agent = QLearningAgent(actions=[0, 1, 2, 3])

# SAFE LOAD
try:
    load_model(agent)
    print("Model loaded")
except Exception as e:
    print("No model found:", e)


# SIMULATION
def simulate(steps):
    try:
        steps = max(1, min(int(steps), 50))
        result = run_simulation(agent, steps)

        # SAFE STRING OUTPUT
        output_lines = []
        for r in result["results"]:
            line = (
                f"Step {r.get('step')} | "
                f"Q:{r.get('queue')} | "
                f"C:{r.get('counters')} | "
                f"A:{r.get('decision')} | "
                f"R:{r.get('reward')}"
            )
            output_lines.append(line)

        output = "\n".join(output_lines)

        summary = f"Total Reward: {result['summary'].get('total_reward', 0)}"

        # SAFE GRAPH
        steps_list = [r.get("step", 0) for r in result["results"]]
        queues = [r.get("queue", 0) for r in result["results"]]

        plt.figure()
        plt.plot(steps_list, queues)
        plt.title("Queue Trend")
        plt.xlabel("Step")
        plt.ylabel("Queue")

        fig = plt.gcf()
        plt.close()

        return output, summary, fig

    except Exception as e:
        return "Error occurred", str(e), None


# UI (MINIMAL SAFE)
with gr.Blocks() as demo:
    gr.Markdown("## Canteen RL System")

    steps_input = gr.Slider(5, 50, value=20, step=1, label="Steps")

    run_btn = gr.Button("Run Simulation")

    output_box = gr.Textbox(label="Simulation Output", lines=15)
    summary_box = gr.Textbox(label="Summary")
    graph = gr.Plot(label="Queue Graph")

    run_btn.click(
        fn=simulate,
        inputs=steps_input,
        outputs=[output_box, summary_box, graph]
    )


# LAUNCH (FINAL SAFE)
demo.launch()