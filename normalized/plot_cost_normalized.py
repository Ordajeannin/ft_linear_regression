import json
import matplotlib.pyplot as plt


def load_cost(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    cost = load_cost("files/cost_history_normalized.json")

    plt.figure(figsize=(10, 6))

    plt.plot(cost)

    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost function - normalized training")

    #permet de mieux observer la convergence du coût, surtout si elle est rapide au début
    #plt.yscale("log")

    plt.grid()

    output_path = "files/cost_normalized.png"
    plt.savefig(output_path, dpi=300)

    plt.show()


if __name__ == "__main__":
    main()