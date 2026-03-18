# Linear Regression — Normalized Version

## Overview

This implementation trains a simple linear regression model using gradient descent, but **with normalized data**.

The model remains the same:

```text
price = theta0 + theta1 * mileage
```

However, the input data is transformed before training.

---

## Why Normalize Data

Raw data can have very different scales:

* mileage: up to ~250000
* price: up to ~10000

This creates problems:

* very large gradients
* unstable training
* need for extremely small learning rates

Normalization fixes this by rescaling data.

---

## Normalization Formula

Each value is transformed using:

```text
x_normalized = (x - mean) / std
```

* `mean`: average of the dataset
* `std`: standard deviation

After normalization:

* mean ≈ 0
* std ≈ 1
* values are centered and scaled

---

## Effect of Normalization

After transformation:

* mileage values become small (≈ -2 to +2)
* gradients become well-scaled
* learning becomes stable

This allows:

* larger learning rates
* faster convergence
* fewer iterations

---

## Constants

```python
LEARNING_RATE = 0.01
ITERATIONS = 1000
```

Compared to non-normalized:

* learning rate is much larger
* fewer issues with divergence

---

## Data Preparation

Before training, you must:

1. Compute mean and std
2. Normalize mileages
3. (optionally) normalize prices

Example:

```python
mean = sum(mileages) / len(mileages)
std = sqrt(...)
```

Then:

```python
normalized = (x - mean) / std
```

---

## Prediction Function

Same as before:

```python
def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage
```

Important: here `mileage` is normalized.

---

## Cost Function

Same formula:

```text
J(theta0, theta1) = (1 / 2m) * Σ(prediction - real_price)^2
```

But now:

* inputs are normalized
* cost decreases more smoothly

---

## Training (Gradient Descent)

The algorithm is identical:

```python
theta0 -= learning_rate * gradient0
theta1 -= learning_rate * gradient1
```

But behavior is very different:

### Without normalization:

* slow
* unstable

### With normalization:

* fast
* smooth convergence

---

## Cost Evolution

With normalized data:

* cost decreases steadily
* curve is smooth
* convergence happens quickly

With non-normalized data:

* cost may stagnate or explode
* requires many iterations

---

## Important: Using the Model

After training, the model expects **normalized input**.

So when predicting:

```python
normalized_mileage = (mileage - mean) / std
price = theta0 + theta1 * normalized_mileage
```

If you skip this step:

* predictions will be completely wrong

---

## Saving Additional Parameters

You must also save:

* mean
* std

Example:

```json
{
    "theta0": ...,
    "theta1": ...,
    "mean": ...,
    "std": ...
}
```

Otherwise you cannot correctly use the model later.

---

## Advantages

* faster training
* stable gradient descent
* easier hyperparameter tuning
* fewer iterations needed

---

## Limitations

* requires preprocessing step
* must store normalization parameters
* prediction requires normalization

---

## Conclusion

Normalization transforms the data so that gradient descent behaves properly.

It is not optional in practice:
it is essential for efficient and stable learning.

Compared to the non-normalized version, it provides:

* faster convergence
* better numerical stability
* simpler tuning of learning rate
