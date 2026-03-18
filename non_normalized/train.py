import csv
import json

DATASET_PATH = "../data.csv"
THETAS_PATH = "files/thetas.json"

LEARNING_RATE = 0.0000000001
#ITERATIONS = 100
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

#les valeurs sont tellement enormes que, non normalisee, ca donne n importe quoi
# a la premiere iterations :
# error = prediction - price = 0 - price
# -> erreur negatives
# sum_error_mileage += error * mileage = negatif
# theta1 -= negatif, donc theta1 devient positif -> ax + b avec a positif
def train(mileages, prices, learning_rate, iterations):
    theta0 = 0.0
    theta1 = 0.0
    m = len(mileages)

    cost_history = []

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

    return theta0, theta1, cost_history


def save_cost_history(cost_history, path):
    import json
    with open(path, "w") as f:
        json.dump(cost_history, f)

def save_model(theta0, theta1, path):
    with open(path, "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f, indent=4)


def main():
    mileages, prices = load_data(DATASET_PATH)
    theta0, theta1, cost_history = train(mileages, prices, LEARNING_RATE, ITERATIONS)
    r2 = compute_r2_score(mileages, prices, theta0, theta1)

    save_model(theta0, theta1, THETAS_PATH)
    save_cost_history(cost_history, "files/cost_history.json")

    print("Training complete (non-normalized).")
    print(f"theta0 = {theta0}")
    print(f"theta1 = {theta1}")
    print(f"final cost = {cost_history[-1]}")
    print(f"R² score = {r2}")


if __name__ == "__main__":
    main()