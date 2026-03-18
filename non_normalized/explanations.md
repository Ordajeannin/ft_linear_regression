# Linear Regression — Non Normalized Version

## Overview

This implementation trains a simple linear regression model using gradient descent to predict car prices from mileage.

The model follows the equation:

```
price = theta0 + theta1 * mileage
```

* `theta0`: intercept
* `theta1`: slope

The goal is to find values of `theta0` and `theta1` that minimize the prediction error.

---

## Constants

```python
LEARNING_RATE = 0.0000000001
ITERATIONS = 100
```

* `LEARNING_RATE`: step size for gradient descent
* `ITERATIONS`: number of updates

Since the data is not normalized, mileage values are very large (e.g. 200000), which forces us to use a very small learning rate to avoid divergence.

---

## Data Loading

```python
def load_data(path):
```

Reads the dataset and returns two lists:

* `mileages`
* `prices`

Each value is converted to `float`.

---

## Prediction Function

```python
def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage
```

Computes the predicted price using the current model.

---

## Cost Function

```python
def compute_cost(mileages, prices, theta0, theta1):
```

Formula:

```
J(theta0, theta1) = (1 / 2m) * Σ(prediction - real_price)^2
```

Steps:

1. Compute prediction
2. Compute error
3. Square error
4. Average over dataset

Purpose:

* Measure how wrong the model is
* Guide gradient descent

---

## R² Score

```python
def compute_r2_score(...)
```

Formula:

```
R² = 1 - (SSres / SStotal)
```

Interpretation:

* `1` → perfect model
* `~0` → useless model
* `<0` → worse than mean

---

## Training (Gradient Descent)

```python
def train(mileages, prices, learning_rate, iterations):
```

### Initialization

```
theta0 = 0
theta1 = 0
```

---

### Loop

For each iteration:

#### 1. Compute errors

```
error = prediction - real_price
```

#### 2. Accumulate gradients

```
sum_error += error
sum_error_mileage += error * mileage
```

#### 3. Compute updates

```
tmp_theta0 = learning_rate * (sum_error / m)
tmp_theta1 = learning_rate * (sum_error_mileage / m)
```

#### 4. Update parameters

```
theta0 -= tmp_theta0
theta1 -= tmp_theta1
```

#### 5. Save cost

```
cost_history.append(...)
```

---

## Why Non-Normalized is Problematic

The dataset uses raw values:

* mileage: up to ~250000
* price: up to ~10000

### Consequences

* gradients become very large
* requires extremely small learning rate
* slow convergence
* unstable training

---

## Saving Results

### Model

```python
save_model(theta0, theta1)
```

Stored as:

```
{
    "theta0": ...,
    "theta1": ...
}
```

---

### Cost History

```python
save_cost_history(cost_history)
```

Used to visualize convergence.

---

## Main Flow

```
load data
→ train model
→ compute R²
→ save model
→ save cost history
→ print results
```

---

## Limitations

* very slow learning
* sensitive to data scale
* difficult to tune learning rate
* may require many iterations

---

## Conclusion

This version works but is inefficient due to the scale of the data.

It highlights why normalization is important:

* faster convergence
* more stable training
* easier tuning
