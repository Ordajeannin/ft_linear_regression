import json
import matplotlib.pyplot as plt


def load_cost(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    cost_normal = load_cost("files/cost_history.json")

    plt.figure(figsize=(10, 6))

    plt.plot(cost_normal, label="Non-normalized")

    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost function during training")
    plt.legend()


    plt.yscale("log")
    output_path = "files/cost_non_normalized.png"
    plt.savefig(output_path, dpi=300)

    plt.show()


if __name__ == "__main__":
    main()