import csv
import json
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

DATASET_PATH = "../data.csv"
THETAS_PATH = "files/thetas_normalized.json"
COST_HISTORY_PATH = "files/cost_history_normalized.json"
THETA_HISTORY_PATH = "files/theta_history_normalized.json"
ANIMATION_PATH = "files/animation_normalized.gif"

LEARNING_RATE = 0.01
ITERATIONS = 1000
ANIMATION_FRAMES = 150
ANIMATION_INTERVAL_MS = 80
ANIMATION_FPS = 20


# lis le csv en sautant la premiere ligne pour eviter d interpreter mileages et prices
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


def mean(values):
    return sum(values) / len(values)


# calcul de l'ecart-type
def std(values, avg):
    variance = sum((x - avg) ** 2 for x in values) / len(values)
    return variance ** 0.5


# normalisation standard des donnees, mean = 0 et std = 1
def normalize(values):
    avg = mean(values)
    deviation = std(values, avg)

    if deviation == 0:
        return [0 for _ in values], avg, deviation

    normalized = [(x - avg) / deviation for x in values]
    return normalized, avg, deviation


# normalise une seule valeur avec la moyenne et l'ecart-type appris pendant le train
def normalize_value(value, avg, deviation):
    if deviation == 0:
        return 0.0
    return (value - avg) / deviation


# tentative de l'algo d'estimer un prix
def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


# J(theta0, theta1) = 1 / (2m) * somme((prediction - reel)^2)
def compute_cost(mileages, prices, theta0, theta1):
    m = len(mileages)
    total = 0.0

    for i in range(m):
        prediction = estimate_price(mileages[i], theta0, theta1)
        error = prediction - prices[i]
        total += error ** 2

    return total / (2 * m)


# coef de determination, comparaison du modele avec un naif "prix moyen"
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


# meme train que ton fichier, mais on garde aussi l'historique des thetas
# pour pouvoir animer le placement progressif de la droite
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
        theta_history.append({
            "theta0": theta0,
            "theta1": theta1,
        })

    return theta0, theta1, cost_history, theta_history


def save_model(theta0, theta1, mean_mileage, std_mileage, path):
    data = {
        "theta0": theta0,
        "theta1": theta1,
        "mean_mileage": mean_mileage,
        "std_mileage": std_mileage,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_cost_history(cost_history, path):
    with open(path, "w") as f:
        json.dump(cost_history, f, indent=4)


def save_theta_history(theta_history, path):
    with open(path, "w") as f:
        json.dump(theta_history, f, indent=4)


# reduit le nombre de frames si on a beaucoup d'iterations,
# pour eviter un gif trop lourd

def sample_theta_history(theta_history, max_frames):
    if len(theta_history) <= max_frames:
        return theta_history

    sampled = []
    last_index = len(theta_history) - 1

    for frame in range(max_frames):
        index = int(frame * last_index / (max_frames - 1))
        sampled.append(theta_history[index])

    return sampled


# animation de la droite sur les donnees d'origine
# le modele est entraine sur mileage normalise,
# mais l'affichage se fait avec les vrais mileages pour etre plus lisible
def create_animation(
    mileages,
    prices,
    theta_history,
    mean_mileage,
    std_mileage,
    output_path,
    max_frames=ANIMATION_FRAMES,
    interval_ms=ANIMATION_INTERVAL_MS,
    fps=ANIMATION_FPS,
):
    sampled_history = sample_theta_history(theta_history, max_frames)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(mileages, prices)
    ax.set_title("Regression line during gradient descent (normalized mileage)")
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Price")

    x_min = min(mileages)
    x_max = max(mileages)
    y_min = min(prices)
    y_max = max(prices)
    y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 1.0

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    line, = ax.plot([], [], linewidth=2)
    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    x_line = [x_min, x_max]

    def update(frame_index):
        state = sampled_history[frame_index]
        theta0 = state["theta0"]
        theta1 = state["theta1"]

        normalized_x_line = [normalize_value(x, mean_mileage, std_mileage) for x in x_line]
        y_line = [estimate_price(x, theta0, theta1) for x in normalized_x_line]

        line.set_data(x_line, y_line)

        original_iteration = int(frame_index * (len(theta_history) - 1) / (len(sampled_history) - 1)) + 1 if len(sampled_history) > 1 else 1
        info_text.set_text(
            f"iteration: {original_iteration}/{len(theta_history)}\n"
            f"theta0: {theta0:.2f}\n"
            f"theta1: {theta1:.2f}"
        )

        return line, info_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(sampled_history),
        interval=interval_ms,
        blit=True,
        repeat=False,
    )

    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.normpath(os.path.join(base_dir, DATASET_PATH))
    thetas_path = os.path.join(base_dir, THETAS_PATH)
    cost_history_path = os.path.join(base_dir, COST_HISTORY_PATH)
    theta_history_path = os.path.join(base_dir, THETA_HISTORY_PATH)
    animation_path = os.path.join(base_dir, ANIMATION_PATH)

    os.makedirs(os.path.dirname(thetas_path), exist_ok=True)

    mileages, prices = load_data(dataset_path)
    normalized_mileages, mean_mileage, std_mileage = normalize(mileages)

    theta0, theta1, cost_history, theta_history = train(
        normalized_mileages,
        prices,
        LEARNING_RATE,
        ITERATIONS,
    )

    r2 = compute_r2_score(normalized_mileages, prices, theta0, theta1)

    save_model(theta0, theta1, mean_mileage, std_mileage, thetas_path)
    save_cost_history(cost_history, cost_history_path)
    save_theta_history(theta_history, theta_history_path)

    create_animation(
        mileages,
        prices,
        theta_history,
        mean_mileage,
        std_mileage,
        animation_path,
    )

    print("Training complete (normalized).")
    print(f"theta0 = {theta0}")
    print(f"theta1 = {theta1}")
    print(f"mean_mileage = {mean_mileage}")
    print(f"std_mileage = {std_mileage}")
    print(f"final cost = {cost_history[-1]}")
    print(f"R² score = {r2}")
    print(f"Animation saved to: {animation_path}")
    print(f"Theta history saved to: {theta_history_path}")


if __name__ == "__main__":
    main()
