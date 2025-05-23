# Network Class Documentation

## Introduction

La classe `Network` permet de créer, entraîner et évaluer des réseaux de neurones. Elle offre plusieurs méthodes pour ajouter des couches, définir des fonctions de perte, entraîner le réseau, et afficher des statistiques d'entraînement.

## Méthodes Disponibles

### `__init__(self, loss='mse')`

Initialise un nouvel objet `Network`.

- **Paramètres** :
  - `loss` : Nom de la fonction de perte (par défaut `mse`).

### `load(self, filename)`

Charge les paramètres d'un `Network` déjà entraîné.

- **Paramètres**:
  - `filename` : Nom du fichier contenant les paramètres sauvegardés.

### `save(self, filename)`

Sauvegarde les paramètres d'un `Network` déjà entraîné.

- **Paramètres**:
  - `filename` : Nom du fichier dans lequel sauvegarder les paramètres.

### `add(self, layer)`

Ajoute une couche au réseau.

- **Paramètres** :
  - `layer` : Objet de type `Layer` à ajouter au réseau.

### `fit(self, x_train, y_train, epochs, learning_rate, silent=False, eval=True, threshold=None, patience=0)`

Entraîne le réseau sur les données d'entraînement.

- **Paramètres** :
  - `x_train` : Données d'entrée pour l'entraînement.
  - `y_train` : Données de sortie attendues pour l'entraînement.
  - `epochs` : Nombre d'époques d'entraînement.
  - `learning_rate` : Taux d'apprentissage.
  - `silent` : Si `True`, n'affiche pas les statistiques d'entraînement (par défaut `False`).
  - `eval` : Si `True`, évalue le réseau après chaque époque (par défaut `True`).
  - `threshold` : Seuil pour l'arrêt anticipé.
  - `patience` : Nombre d'époques à attendre avant l'arrêt anticipé si le seuil n'est pas atteint.

### `predict(self, input_data)`

Prédit la sortie pour les données d'entrée données.

- **Paramètres** :
  - `input_data` : Données d'entrée pour la prédiction.

- **Retourne** :
  - Liste des prédictions pour chaque entrée.

### `evaluate(self, x_test, y_test, silent=True)`

Évalue le réseau sur les données de test.

- **Paramètres** :
  - `x_test` : Données d'entrée pour le test.
  - `y_test` : Données de sortie attendues pour le test.
  - `silent` : Si `True`, n'affiche pas les statistiques d'évaluation (par défaut `True`).

- **Retourne** :
  - Précision du réseau sur les données de test.

### `clear_logs(self)`

Efface les statistiques d'entraînement.

### `summary(self)`

Affiche un résumé du réseau.

### `disp_loss_graph(self)`

Affiche le graphique des erreurs d'entraînement.

### `disp_accuracy_graph(self)`

Affiche le graphique des précisions d'entraînement.

### `disp_loss_accuracy_graph(self)`

Affiche les graphiques des erreurs et des précisions d'entraînement.

### `show(self)`

Affiche les graphiques.

### `confusion_matrix(self, x_test, y_test)`

Génère la matrice de confusion pour le réseau.


- **Paramètres** :
  - `x_test` : Données d'entrée pour le test.
  - `y_test` : Données de sortie attendues pour le test.

- **Retourne** :
  - Matrice de confusion.

## Exemples d'utilisation

### Création et entraînement d'un réseau

```python
import numpy as np
from neural_network.FCLayer import FCLayer
from neural_network.ActivationLayer import ActivationLayer
from neural_network.ActivationFunc import *
from neural_network.LossesFunc import *
from neural_network.Network import Network

# Données d'entraînement
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Création du réseau
net = Network('mse')
net.add(FCLayer(2, 3))
net.add(ActivationLayer('tanh'))
net.add(FCLayer(3, 1))
net.add(ActivationLayer('sigmoid'))

# Entraîner le réseau
net.fit(x_train, y_train, epochs=2000, learning_rate=0.1)

# Afficher un résumé du réseau
net.summary()

# Afficher les graphiques des erreurs et des précisions
net.disp_loss_accuracy_graph()
net.show()

# Prédire les sorties pour les données d'entraînement
out = net.predict(x_train)
print(out)

# Générer la matrice de confusion
conf_matrix = net.confusion_matrix(x_train, y_train)
print(conf_matrix)