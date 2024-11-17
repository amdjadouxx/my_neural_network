import numpy as np
import os
import time
from neural_network.Network import Network
from neural_network.FCLayer import FCLayer
from neural_network.ActivationLayer import ActivationLayer
from neural_network.DropoutLayer import DropoutLayer

# Génération de données synthétiques
np.random.seed(42)
num_samples = 1000
house_size = np.random.rand(num_samples, 1) * 100  # Taille en mètres carrés

# Normalisation des données (Min-Max scaling entre 0 et 1)
house_size = house_size / 100.0

# Classification des maisons comme petite, moyenne ou grande
labels = np.zeros((num_samples, 3))
for i in range(num_samples):
    if house_size[i] < 0.33:  # Petite (moins de 33%)
        labels[i, 0] = 1
    elif house_size[i] < 0.66:  # Moyenne (entre 33% et 66%)
        labels[i, 1] = 1
    else:  # Grande (plus de 66%)
        labels[i, 2] = 1

x_train = house_size
y_train = labels

def save():
    start_time = time.time()
    net = Network('mse')  # Fonction de perte (Mean Squared Error)
    net.add(FCLayer(1, 5))  # Couche entièrement connectée
    net.add(ActivationLayer('relu'))  # Activation ReLU
    net.add(DropoutLayer(0.2))  # Dropout (20% des neurones désactivés)
    net.add(FCLayer(5, 3))  # Dernière couche pour les 3 classes
    net.add(ActivationLayer('softmax'))  # Softmax pour la classification multi-classe

    # Entraînement du réseau
    net.fit(x_train, y_train, epochs=1000, learning_rate=0.01, silent=False, threshold=0.01, patience=10)
    net.save('house_classification.pkl')
    print("--- %s seconds ---" % (time.time() - start_time))
    return net

def prediction(net):
    # Données de test (normalisées comme les données d'entraînement)
    test_data = np.array([[50], [100], [10]]) / 100.0
    predictions = net.predict(test_data)
    for i, pred in enumerate(predictions):
        category = np.argmax(pred)
        if category == 0:
            size = "Small"
        elif category == 1:
            size = "Medium"
        else:
            size = "Large"
        print(f"House {i+1}: Predicted size: {size}")

if __name__ == "__main__":
    if os.path.exists('house_classification.pkl'):
        net = Network()
        net.load('house_classification.pkl')
    else:
        net = save()

    net.summary()
    prediction(net)
