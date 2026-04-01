import matplotlib.pyplot as plt


def plot_rewards(rewards):
    if not rewards:
        print("No data to plot.")
        return

    plt.figure()

    plt.plot(rewards, label="Episode Reward")

    # Moving average (smoother curve)
    window = 10
    if len(rewards) >= window:
        avg = [
            sum(rewards[i:i + window]) / window
            for i in range(len(rewards) - window + 1)
        ]
        plt.plot(range(window - 1, len(rewards)), avg, label="Moving Avg (10)")

    plt.title("Training Progress")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    plt.legend()
    plt.grid()

    plt.show()