import csv
import json
import matplotlib.pyplot as plt

DATASET_PATH = "../data.csv"
THETAS_PATH = "files/thetas_normalized.json"


def load_data(path):
    mileages = []
    prices = []

    with open(path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            mileages.append(float(row[0]))
            prices.append(float(row[1]))

    return mileages, prices


def load_model(path):
    with open(path, "r") as f:
        data = json.load(f)

    return (
        data["theta0"],
        data["theta1"],
        data["mean_mileage"],
        data["std_mileage"],
    )


def normalize_mileage(mileage, mean_mileage, std_mileage):
    if std_mileage == 0:
        return 0.0
    return (mileage - mean_mileage) / std_mileage


def estimate_price(mileage, theta0, theta1, mean_mileage, std_mileage):
    normalized = normalize_mileage(mileage, mean_mileage, std_mileage)
    return theta0 + theta1 * normalized


def main():
    mileages, prices = load_data(DATASET_PATH)
    theta0, theta1, mean_mileage, std_mileage = load_model(THETAS_PATH)

    pairs = sorted(zip(mileages, prices), key=lambda pair: pair[0])
    sorted_mileages = [pair[0] for pair in pairs]

    predicted_prices = [
        estimate_price(mileage, theta0, theta1, mean_mileage, std_mileage)
        for mileage in sorted_mileages
    ]

    plt.figure(figsize=(10, 6))
    plt.scatter(mileages, prices, label="Real data")
    plt.plot(sorted_mileages, predicted_prices, label="Regression line")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.title("Linear regression - normalized")
    plt.legend()

    output_path = "files/plot_normalized.png"
    plt.savefig(output_path, dpi=300)

    plt.show()


if __name__ == "__main__":
    main()