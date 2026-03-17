import json
import os


THETAS_PATH = "thetas_normalized.json"


def load_model(path):
    if not os.path.exists(path):
        return 0.0, 0.0, 0.0, 1.0

    with open(path, "r") as f:
        data = json.load(f)

    theta0 = data.get("theta0", 0.0)
    theta1 = data.get("theta1", 0.0)
    mean_mileage = data.get("mean_mileage", 0.0)
    std_mileage = data.get("std_mileage", 1.0)

    return theta0, theta1, mean_mileage, std_mileage


def normalize_mileage(mileage, mean_mileage, std_mileage):
    if std_mileage == 0:
        return 0.0
    return (mileage - mean_mileage) / std_mileage


def estimate_price(normalized_mileage, theta0, theta1):
    return theta0 + theta1 * normalized_mileage


def main():
    theta0, theta1, mean_mileage, std_mileage = load_model(THETAS_PATH)

    try:
        mileage = float(input("Enter mileage: "))
        if mileage < 0:
            print("Mileage cannot be negative.")
            return

        normalized_mileage = normalize_mileage(mileage, mean_mileage, std_mileage)
        price = estimate_price(normalized_mileage, theta0, theta1)

        if price < 0:
            price = 0.0

        print(f"Estimated price: {price:.2f}")
    except ValueError:
        print("Invalid input. Please enter a valid number.")


if __name__ == "__main__":
    main()