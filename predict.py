import json
import os

THETAS_PATH = "thetas.json"


def load_thetas(path):
    if not os.path.exists(path):
        return 0.0, 0.0

    with open(path, "r") as f:
        data = json.load(f)

    return data.get("theta0", 0.0), data.get("theta1", 0.0)


def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


def main():
    theta0, theta1 = load_thetas(THETAS_PATH)

    try:
        mileage = float(input("Enter mileage: "))
        if mileage < 0:
            print("Mileage cannot be negative.")
            return

        price = estimate_price(mileage, theta0, theta1)
        if price < 0:
            price = 0.0

        print(f"Estimated price: {price:.2f}")
    except ValueError:
        print("Invalid input. Please enter a valid number.")


if __name__ == "__main__":
    main()