# 🌍 Projet Big Data – Traitement distribué d’images de fruits sur le cloud

## 🚀 Contexte du projet

Dans le cadre de ma mission en tant que Data Scientist chez **Fruits!**, une jeune start-up AgriTech, j’ai été chargé de construire les **premières briques d’une architecture Big Data** sur le cloud AWS, afin de traiter à grande échelle des images de fruits.

Ce traitement s’inscrit dans le développement d’une application mobile éducative, destinée à sensibiliser le public à la **biodiversité des fruits** et à amorcer un futur moteur de reconnaissance d’images pour les cueilleurs automatisés.

---

## 🧱 Objectifs techniques

- Reprendre et compléter le **notebook PySpark** d’un précédent alternant
- Implémenter une **réduction de dimension (PCA)** distribuée sur Spark
- Diffuser les **poids d’un modèle TensorFlow** sur les nœuds du cluster via Spark `broadcast`
- Déployer et tester une architecture **EMR (Elastic Map Reduce)** sur AWS
- Respecter les contraintes **RGPD** en choisissant une région de cloud européenne
- Documenter et démontrer la chaîne de traitement de manière claire et reproductible

---

## ⚙️ Environnement Cloud & Big Data

- **Cloud provider** : AWS  
- **Stockage** : S3 (datasets & résultats)  
- **Traitement distribué** : PySpark sur cluster EMR  
- **Sécurité & coûts** :
  - Région : `eu-west-1` (Irlande) pour conformité RGPD
  - Cluster EMR lancé temporairement uniquement pour les tests (<10€)
  - Développement et debug réalisés en local pour optimiser les coûts

---

## 📁 Contenu du projet

- `notebook_pyspark_fruits.ipynb` : traitement distribué avec PySpark (lecture des images, PCA, broadcast)
- `script_broadcast_tensorflow.py` : diffusion des poids du modèle pour l’inférence en parallèle
- `emr_setup.md` : guide étape par étape de création et configuration d’un cluster EMR
- `README.md` : ce document

---

## 🧪 Étapes du traitement PySpark

1. Chargement des images de fruits depuis AWS S3
2. Transformation et vectorisation des images
3. Réduction de dimension via **PCA Spark MLlib**
4. **Broadcast des poids TensorFlow** et simulation d’inférence distribuée
5. Sauvegarde des résultats dans S3

---

## 📦 Téléchargement du projet

Le projet complet (scripts, notebooks, documentation, données d’exemple) est disponible ici 
---

## 🛡️ RGPD & bonnes pratiques

- Traitement conforme RGPD (région AWS : Europe)
- Données anonymisées, aucun usage personnel ou sensible
- Coût maîtrisé du cloud (instance arrêtée hors démonstration)
- Scripts commentés, reproductibles, conformes aux standards PEP8

---

## 💬 Retour critique

L’approche EMR offre une **bonne scalabilité** pour des projets de vision par ordinateur en croissance. Toutefois, la diffusion des modèles lourds en TensorFlow reste coûteuse et doit être optimisée. Une piste d’amélioration serait l’utilisation de **Databricks** pour simplifier l’intégration et améliorer la collaboration.

---

## 👤 Auteur

**Noureddine YAMOUNI**  
Ingénieur en Intelligence Artificielle – Data Scientist  
📫 Contact : yamouninoureddine99@gmail.com
