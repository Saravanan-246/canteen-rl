import os


def save_model(agent, path="q_table.json"):
    try:
        agent.save(path)
        print(f"Model saved to {path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(agent, path="q_table.json"):
    if not os.path.exists(path):
        raise FileNotFoundError("Model file not found")

    try:
        agent.load(path)
        print(f"Model loaded from {path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise Exception("Failed to load model")