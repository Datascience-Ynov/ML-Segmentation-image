# Segmentation d'Images avec Apprentissage Non Supervisé

## Contexte du Projet

Ce projet vise à réaliser une segmentation d'images en utilisant des algorithmes d'apprentissage non supervisé. La segmentation d'images consiste à diviser une image en plusieurs régions ou clusters, où chaque région partage des caractéristiques similaires (couleur, texture, etc.). L'objectif est de tester et comparer plusieurs algorithmes de clustering pour identifier les meilleures approches pour différents types d'images.

Le projet inclut également la création d'une **interface graphique intuitive** pour permettre aux utilisateurs de visualiser les résultats de la segmentation en temps réel.

## Objectifs

1. **Préparation des données** : Transformer et normaliser les images pour les rendre exploitables par les algorithmes de clustering.
2. **Entraînement des modèles** : Appliquer plusieurs algorithmes de clustering pour segmenter les images.
3. **Interface graphique** : Développer une interface utilisateur interactive avec **Streamlit** pour visualiser les résultats.
4. **Comparaison des algorithmes** : Évaluer et comparer les performances des différents algorithmes de clustering.
5. **Visualisation des résultats** : Générer des visualisations pour mieux comprendre les clusters et les segments obtenus.

## Technologies Utilisées

- **Langage** : Python
- **Bibliothèques** :
  - **Scikit-learn** : Pour l'implémentation des algorithmes de clustering.
  - **OpenCV** : Pour le traitement et la manipulation des images.
  - **Streamlit** : Pour la création de l'interface graphique.
  - **Pandas** : Pour la manipulation des données.
  - **Matplotlib** : Pour la visualisation des résultats.
  - **PCA (Analyse en Composantes Principales)** : Pour la réduction de dimensionnalité.
- **Algorithmes de Clustering Testés** :
  - **KMeans**
  - **k_means**
  - **K-Means++**
  - **DBSCAN**
  - **HDBSCAN**
  - **Agglomerative Clustering**
  - **Affinity Propagation**
  - **Mean Shift**
  - **MiniBatch KMeans**
  - **Spectral Clustering**
  - **Spectral BiClustering**
  - **Fuzzy C-Means**
  - **Birch**
  - **OPTICS**
  - **Cluster-optics-dbscan**
  - **SOM (Self-Organizing Maps)**
  - **Bissecting KMeans**
  - **Feature Agglomeration**
  - **Estimate Bandwidth**

## Étapes du Projet

### 1. Préparation des Données
- Chargement des images avec **OpenCV**.
- Conversion des images en tableaux de pixels (forme : `(n_pixels, 3)` pour les canaux RGB).
- Normalisation des valeurs des pixels (mise à l'échelle entre 0 et 1).

### 2. Entraînement des Modèles
- Application des algorithmes de clustering sur les données d'images.
- Utilisation de techniques comme **PCA** pour réduire la dimensionnalité des données si nécessaire.
- Optimisation des hyperparamètres pour chaque algorithme (par exemple, nombre de clusters, distance minimale, etc.).

### 3. Interface Graphique
- Développement d'une interface utilisateur avec **Streamlit** pour :
  - Télécharger une image.
  - Choisir un algorithme de clustering.
  - Afficher l'image segmentée.
  - Enregistrer l'image segmentée (optionnel).
  - Visualiser les clusters en 3D (optionnel).

### 4. Évaluation des Modèles
- Visualisation des résultats de segmentation (images segmentées avec des couleurs différentes pour chaque cluster).
- Comparaison des performances des algorithmes en termes de temps d'exécution et de qualité de segmentation.

### 5. Visualisation des Résultats
- Génération de graphiques pour comparer les performances des algorithmes.
- Affichage des images segmentées avec des légendes pour chaque cluster.

## Résultats Obtenus

- **Meilleurs Algorithmes** : [K-Means, Bissecting KMeans]
- **Comparaison des Performances** :
  - **K-Means** : Rapide et efficace pour les images simples.
  - **DBSCAN** : Performant pour les images avec des textures complexes.
  - **Fuzzy C-Means** : Utile pour les images avec des frontières floues entre les clusters.
  - **SOM** : Performant pour les images de grande taille.

## Comment Utiliser ce Projet

1. **Cloner le dépôt** :
   ```bash
   git clone git@github.com:Datascience-Ynov/ML-Segmentation-image.git
   cd Segmentation-Images
2. **Lancer l'application Streamlit** :
   ```bash
   streamlit run Streamlit_App.py
3. **Utiliser l'interface graphique** :
   - **Télécharger une image.**
   - **Choisir un algorithme de clustering.**
   - **Visualiser l'image segmentée et les résultats.**


### Auteur
Nom : **AMOUSSA Mourad**

Contact : [mourad1.amoussa@outlook.com]

LinkedIn : [www.linkedin.com/in/mourad-amoussa]

GitHub : [https://github.com/Mourad2511]
