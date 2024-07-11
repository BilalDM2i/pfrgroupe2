1. Arbre de Décision

python

from sklearn.tree import DecisionTreeClassifier

# Création du pipeline complet avec un modèle d'arbre de décision
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions
predictions = model.predict(X_test)

2. Forêt Aléatoire

python

from sklearn.ensemble import RandomForestClassifier

# Création du pipeline complet avec un modèle de forêt aléatoire
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions
predictions = model.predict(X_test)

3. Support Vector Machines (SVM)

python

from sklearn.svm import SVC

# Création du pipeline complet avec un modèle SVM
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions
predictions = model.predict(X_test)

4. Réseau de Neurones (MLPClassifier)

python

from sklearn.neural_network import MLPClassifier

# Création du pipeline complet avec un modèle de réseau de neurones
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(100, ), max_iter=300))
])

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions
predictions = model.predict(X_test)

5. K-Nearest Neighbors (KNN)

python

from sklearn.neighbors import KNeighborsClassifier

# Création du pipeline complet avec un modèle KNN
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions
predictions = model.predict(X_test)