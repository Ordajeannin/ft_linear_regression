# ft_linear_regression

## Introduction

Ce projet a pour objectif d’implémenter une **régression linéaire simple** en Python, en utilisant l’algorithme de **descente de gradient**.

L’idée est d’apprendre à une machine à prédire une valeur (le prix d’une voiture) en fonction d’une autre (son kilométrage), à partir d’un dataset.

---

## Objectif

Trouver les meilleurs paramètres `theta0` et `theta1` pour modéliser la relation :

```
price ≈ theta0 + theta1 * mileage
```

Une fois entraîné, le modèle permet de prédire un prix pour n’importe quel kilométrage.

---

## ⚙️ Fonctionnement

### 1. Chargement des données

Le dataset est lu depuis un fichier CSV contenant :

- mileage (km)
- price (€)

---

### 2. Modèle utilisé

On utilise une fonction linéaire :

```
estimate_price = theta0 + theta1 * mileage
```

---

### 3. Fonction de coût

Le modèle est évalué avec une fonction de coût (MSE) :

```
cost = (1/m) * Σ(prediction - réel)²
```

Elle mesure l’erreur moyenne du modèle.

ou 
- m = nombre de donnees
---

### 4. Descente de gradient

Les paramètres sont ajustés progressivement pour minimiser l’erreur :

```
theta0 = theta0 - α * dérivée
theta1 = theta1 - α * dérivée
```

où :

- α = learning rate

---

## 🚀 Utilisation

### Entraîner le modèle

```bash
python3 train.py
```

→ Sauvegarde des paramètres dans `files/thetas.json`

---

### Prédire un prix

```bash
python3 predict.py
```

→ L’utilisateur entre un kilométrage, et le prix estimé est affiché

---

## 📊 Normalisation (important)

Les données peuvent être **normalisées** pour améliorer la convergence :

```
x_normalized = (x - mean) / std
```

### Pourquoi ?

- Évite des valeurs trop grandes (ex: 200000 km)
- Rend la descente de gradient plus stable et rapide
- Permet d’utiliser un learning rate plus raisonnable

Une donnée normalisée a :

- moyenne = 0
- écart-type = 1

---

## ⚠️ Problèmes courants

- Learning rate trop grand → divergence
- Learning rate trop petit → apprentissage lent
- Données non normalisées → instabilité

---

## 🧪 Améliorations possibles

- Normalisation automatique
- Visualisation avec matplotlib
- Sauvegarde de l’historique du coût
- Ajout de plusieurs features (régression multiple)

---

## 🧾 Conclusion

Ce projet introduit les bases du Machine Learning :

- modélisation
- optimisation
- généralisation

C’est une première étape vers des modèles plus complexes.
