# 📊 Dashboard de Credit Scoring – Projet Data Science

## 🎯 Contexte

Dans le cadre d’un projet simulé en environnement professionnel, j’ai intégré l’équipe Data d’une société financière fictive nommée **Prêt à dépenser**, spécialisée dans les crédits à la consommation pour des profils sans historique de prêt.

Après avoir développé un **modèle de scoring crédit**, l’objectif de cette mission a été de concevoir un **dashboard interactif**, clair et pédagogique, destiné aux **chargés de relation client** afin de justifier les décisions d’octroi de crédit auprès des clients.

---

## 🧭 Objectifs fonctionnels

- Visualiser le **score de crédit** et la **probabilité associée** d’un client (accepté/refusé)
- Comparer ses **caractéristiques personnelles** avec la population générale ou un groupe filtré
- Fournir une **interprétation du score** via des explications locales (SHAP) et globales
- Intégrer une **API REST** pour récupérer le score d’un client existant ou simuler un nouveau dossier
- Assurer la **compréhensibilité pour les non-experts**
- Respecter les standards d’**accessibilité** (WCAG)
- Déployer le dashboard sur le **cloud (Streamlit Cloud)** pour accès distant

---

## ⚙️ Stack technique

- **Python** : traitement, visualisation, API calls
- **Streamlit** : interface utilisateur interactive
- **Plotly / Matplotlib / Seaborn** : visualisations
- **XGBoost / Logistic Regression** : modèles de scoring
- **SHAP** : interprétation locale des prédictions
- **API** : Flask + Render pour prédiction en ligne
- **Hébergement** : Streamlit Cloud

---

## 📂 Structure du projet

credit_scoring_dashboard/
│
├── app.py ← Code principal du dashboard Streamlit
├── api_client.py ← Requêtes vers l’API de scoring
├── utils.py ← Fonctions d’analyse et SHAP
├── model/best_model.pkl ← Modèle de scoring entraîné
├── data/application_train.csv ← Données client (échantillon anonymisé)
├── assets/ ← Images, logos, et icônes
└── README.md ← Ce fichier


# 🧠 Mini MoMi – Veille Technique : NLP avec Modern BERT vs TF-IDF

## 🎯 Objectif

Dans le cadre de ma mission de veille technique chez *Prêt à dépenser*, j’ai étudié et comparé une approche classique d’analyse de texte (TF-IDF + régression logistique) avec une méthode plus récente basée sur **Modern BERT**, un modèle pré-entraîné de la famille des Transformers.

L’objectif : tester les performances et l’interprétabilité de **Modern BERT** dans un contexte réel, en l’appliquant aux données clients (commentaires, descriptions) utilisées dans nos projets de scoring ou de classification produit.

---

## 🧪 Méthodes comparées

| Méthode | Description | Modèle | Interprétabilité | Temps d’entraînement |
|--------|-------------|--------|------------------|----------------------|
| **TF-IDF** | Extraction manuelle de features | LogisticRegression | Bonne (coefficients) | Rapide |
| **Modern BERT** | Embeddings contextuels (transformer) | DistilBERT + MLP | Moyenne (via attention/SHAP) | Plus long |

---

## 🔧 Implémentation

- **Données** : texte client anonymisé / jeu de classification public
- **Prétraitement** : nettoyage, tokenisation
- **Modèle Modern BERT** : `distilbert-base-uncased` via HuggingFace Transformers
- **Évaluation** : Accuracy, F1, temps de calcul, explications locales (SHAP)

---

## 📌 Résultats observés

- 📈 BERT offre de meilleures performances sur des textes ambigus ou non structurés.
- ⚙️ TF-IDF reste plus rapide et plus transparent, utile pour des déploiements rapides.
- 🧠 Les représentations de BERT permettent d’intégrer du contexte sémantique inaccessible avec TF-IDF.

## 🔗 Références

- [Devlin et al., 2018 – BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)  
- [HuggingFace Transformers](https://huggingface.co/transformers/)  
- [Papers with Code –]()
