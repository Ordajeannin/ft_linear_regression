import csv
import json
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

DATASET_PATH = "../data.csv"
THETAS_PATH = "files/thetas.json"

LEARNING_RATE = 0.0000000001
ITERATIONS = 500000


def load_data(path):
    mileages = []
    prices = []

    with open(path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            mileage = float(row[0])
            price = float(row[1])
            mileages.append(mileage)
            prices.append(price)

    return mileages, prices


def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


def compute_cost(mileages, prices, theta0, theta1):
    m = len(mileages)
    total = 0.0

    for i in range(m):
        prediction = estimate_price(mileages[i], theta0, theta1)
        error = prediction - prices[i]
        total += error ** 2

    return total / (2 * m)


def compute_r2_score(mileages, prices, theta0, theta1):
    mean_price = sum(prices) / len(prices)
    ss_total = 0.0
    ss_res = 0.0

    for i in range(len(prices)):
        prediction = estimate_price(mileages[i], theta0, theta1)
        ss_res += (prices[i] - prediction) ** 2
        ss_total += (prices[i] - mean_price) ** 2

    if ss_total == 0:
        return 1.0
    return 1 - (ss_res / ss_total)


def train(mileages, prices, learning_rate, iterations):
    theta0 = 0.0
    theta1 = 0.0
    m = len(mileages)

    cost_history = []
    theta_history = []

    for _ in range(iterations):
        sum_error = 0.0
        sum_error_mileage = 0.0

        for i in range(m):
            prediction = estimate_price(mileages[i], theta0, theta1)
            error = prediction - prices[i]

            sum_error += error
            sum_error_mileage += error * mileages[i]

        tmp_theta0 = learning_rate * (sum_error / m)
        tmp_theta1 = learning_rate * (sum_error_mileage / m)

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        cost_history.append(compute_cost(mileages, prices, theta0, theta1))
        theta_history.append((theta0, theta1))

    return theta0, theta1, cost_history, theta_history


def save_cost_history(cost_history, path):
    with open(path, "w") as f:
        json.dump(cost_history, f)


def save_theta_history(theta_history, path):
    data = [
        {"theta0": theta0, "theta1": theta1}
        for theta0, theta1 in theta_history
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_model(theta0, theta1, path):
    with open(path, "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f, indent=4)


def animate_regression(mileages, prices, theta_history, output_path, step=1000):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(mileages, prices, label="Data")
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Price")
    ax.set_title("Gradient Descent - Non-normalized data")

    x_min = min(mileages)
    x_max = max(mileages)
    x_line = [x_min, x_max]

    y_min = min(prices)
    y_max = max(prices)
    margin_y = (y_max - y_min) * 0.1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)

    line, = ax.plot([], [], lw=2, label="Regression line")
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
    ax.legend()

    sampled_theta_history = theta_history[::step]

    if sampled_theta_history[-1] != theta_history[-1]:
        sampled_theta_history.append(theta_history[-1])

    def update(frame):
        theta0, theta1 = sampled_theta_history[frame]
        y_line = [estimate_price(x, theta0, theta1) for x in x_line]

        displayed_iteration = min(frame * step + 1, len(theta_history))

        line.set_data(x_line, y_line)
        text.set_text(
            f"Iteration: {displayed_iteration}\n"
            f"theta0: {theta0:.4f}\n"
            f"theta1: {theta1:.8f}"
        )
        return line, text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(sampled_theta_history),
        interval=50,
        blit=True,
        repeat=False
    )

    animation.save(output_path, writer=PillowWriter(fps=20))
    plt.close(fig)


def main():
    os.makedirs("files", exist_ok=True)

    mileages, prices = load_data(DATASET_PATH)
    theta0, theta1, cost_history, theta_history = train(
        mileages,
        prices,
        LEARNING_RATE,
        ITERATIONS
    )

    r2 = compute_r2_score(mileages, prices, theta0, theta1)

    save_model(theta0, theta1, THETAS_PATH)
    save_cost_history(cost_history, "files/cost_history.json")
    save_theta_history(theta_history, "files/theta_history.json")

    animate_regression(
        mileages,
        prices,
        theta_history,
        "files/animation_non_normalized.gif",
        step=1000
    )

    print("Training complete (non-normalized).")
    print(f"theta0 = {theta0}")
    print(f"theta1 = {theta1}")
    print(f"final cost = {cost_history[-1]}")
    print(f"R² score = {r2}")
    print("Animation saved in files/animation_non_normalized.gif")


if __name__ == "__main__":
    main()