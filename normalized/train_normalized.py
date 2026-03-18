import csv
import json

DATASET_PATH = "../data.csv"
THETAS_PATH = "files/thetas_normalized.json"

LEARNING_RATE = 0.01
ITERATIONS = 1000

#lis le csv en sautant la premiere ligne pour eviter d interpreter mileages et prices
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

#besoin de commenter?
def mean(values):
    return sum(values) / len(values)

#calcul de l'ecart-type
def std(values, avg):
    variance = sum((x - avg) ** 2 for x in values) / len(values)
    return variance ** 0.5

#normalisation standard des donnees, mean = 0 et std = 1
def normalize(values):
    avg = mean(values)
    deviation = std(values, avg)

    if deviation == 0:
        return [0 for _ in values], avg, deviation

    normalized = [(x - avg) / deviation for x in values]
    return normalized, avg, deviation

#tentative de l algo d'estimer un prix
def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

# J(θ0​,θ1​)=1/2m​∑(prediction−reel)^2
def compute_cost(mileages, prices, theta0, theta1):
    m = len(mileages)
    total = 0.0

    for i in range(m):
        prediction = estimate_price(mileages[i], theta0, theta1)
        error = prediction - prices[i]
        total += error ** 2

    return total / (2 * m)

#coef de determination, comparaison du model avec un naif "prix moyen"
# - ss_total = somme des carres total, niveau de desordre naturel
#              "tout ce qu il y a a expliquer"
# - ss_res = somme des carres residuel, erreur restante apres le modele
#            "ce que je n explique pas"
# part expliquee, R^2 = 1 - (ss_res / ss_total)
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
    import json
    with open(path, "w") as f:
        json.dump(cost_history, f)

def main():
    mileages, prices = load_data(DATASET_PATH)
    normalized_mileages, mean_mileage, std_mileage = normalize(mileages)

    theta0, theta1, cost_history = train(
        normalized_mileages,
        prices,
        LEARNING_RATE,
        ITERATIONS,
    )

    r2 = compute_r2_score(normalized_mileages, prices, theta0, theta1)

    save_model(theta0, theta1, mean_mileage, std_mileage, THETAS_PATH)
    save_cost_history(cost_history, "files/cost_history_normalized.json")

    print("Training complete (normalized).")
    print(f"theta0 = {theta0}")
    print(f"theta1 = {theta1}")
    print(f"mean_mileage = {mean_mileage}")
    print(f"std_mileage = {std_mileage}")
    print(f"final cost = {cost_history[-1]}")
    print(f"R² score = {r2}")


if __name__ == "__main__":
    main()