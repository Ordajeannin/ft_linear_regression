import json
import matplotlib.pyplot as plt


def load_cost(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    cost_normal = load_cost("cost_history.json")
    cost_normalized = load_cost("cost_history_normalized.json")

    plt.figure(figsize=(10, 6))

    plt.plot(cost_normal, label="Non-normalized")
    plt.plot(cost_normalized, label="Normalized")

    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost function during training")
    plt.legend()
    plt.yscale("log")

    plt.show()


if __name__ == "__main__":
    main()