# Détection de Fake News avec BERT

## Introduction
This project develops a BERT-based neural network to classify political statements as true or false using the LIAR2 dataset. By integrating textual analysis with multimodal metadata (context, speaker history, and credibility scores), the model achieves an 85% accuracy, significantly outperforming baseline text-only approaches.

## 1. Présentation du modèle

Afin d'améliorer les performances obtenues avec les approches classiques basées sur des représentations de type *bag-of-words* (TF-IDF), nous utilisons un modèle basé sur **BERT (Bidirectional Encoder Representations from Transformers)**.

BERT est un modèle de langage pré-entraîné capable de produire des représentations contextuelles riches d'un texte. Contrairement aux représentations classiques, chaque mot est encodé en tenant compte du contexte global de la phrase.

Dans notre cas, le modèle doit prédire si une déclaration politique est **vraie** ou **fausse**.

Nous combinons :

- une représentation du texte obtenue avec BERT
- des **métadonnées** associées à la déclaration
- un **score de crédibilité de l'orateur**

Ces informations sont fusionnées dans un réseau de neurones final chargé d'effectuer la classification.

---

# 2. Architecture du modèle

L'architecture du modèle peut être résumée par les étapes suivantes :

- Déclaration (texte)
- Tokenizer BERT
- BERT Encoder
- Mean Pooling des tokens
- Concatenation de toutes les features :
  - Représentation du texte (768)
  - Métadonnées (speaker, subject, state, context)
  - Score de crédibilité du speaker
- Classifier (MLP)
- Probabilité Fake / True

Le vecteur final utilisé pour la classification contient donc :

- la représentation du texte produite par BERT
- les embeddings des métadonnées
- le score de crédibilité de l'orateur

Ce vecteur est ensuite passé dans un **réseau de neurones fully connected** pour produire la prédiction finale.

---

# 3. Utilisation des métadonnées

Le dataset contient plusieurs informations supplémentaires associées à chaque déclaration :

- `speaker` : l'orateur
- `subject` : mots clés de la déclaration
- `state_info` : l'état associé
- `context` : le contexte de la déclaration
- `statement` : sujet principale de la déclaration

Ces variables sont **catégorielles**.

Afin de pouvoir les utiliser dans le modèle :

1. chaque catégorie est transformée en indice via un `LabelEncoder`
2. chaque indice est projeté dans un **embedding vectoriel appris**

Chaque métadonnée possède donc son propre espace vectoriel de petite dimension.

Les embeddings de toutes les métadonnées sont ensuite **concaténés avec la représentation du texte** obtenue avec BERT.

Cette approche permet au modèle de prendre en compte le contexte politique ou institutionnel dans lequel la déclaration a été produite.

---

# 4. Score de crédibilité du speaker

Le dataset fournit également un historique des déclarations passées de chaque orateur sous la forme de compteurs :

- `pants_on_fire_counts`
- `false_counts`
- `mostly_false_counts`
- `half_true_counts`
- `mostly_true_counts`
- `true_counts`


Ces informations permettent de calculer un **score de crédibilité** pour chaque speaker.

Nous définissons ce score comme :
credibility = (true_counts + mostly_true + half_true) / total_statements

où :

total_statements =
pants_on_fire + false + mostly_false + half_true + mostly_true + true

Ce score correspond donc à la proportion de déclarations relativement fiables dans l'historique de l'orateur.

Ce score est utilisé comme **feature numérique supplémentaire** et concaténé avec les autres représentations dans le modèle.

L'intuition est que certains orateurs ont historiquement tendance à produire davantage de déclarations fausses ou trompeuses.



---

# 5. Pipeline d'entraînement

Le pipeline d'entraînement du modèle suit les étapes suivantes.

### 1. Préparation des données

- chargement du dataset
- nettoyage minimal des textes
- encodage des métadonnées
- calcul du score de crédibilité du speaker

### 2. Tokenisation

Les déclarations sont transformées en tokens à l'aide du tokenizer de BERT.

Les séquences sont ensuite :

- tronquées à une longueur maximale
- complétées avec du padding si nécessaire

### 3. Encodage avec BERT

Les séquences tokenisées sont passées dans BERT afin d'obtenir une représentation vectorielle du texte.

Nous utilisons un **mean pooling des embeddings de tokens** afin d'obtenir une représentation plus stable de la phrase.

### 4. Fusion des informations

Les informations suivantes sont concaténées :

- représentation du texte (BERT)
- embeddings des métadonnées
- score de crédibilité du speaker

### 5. Classification

Le vecteur résultant est envoyé dans un **réseau fully connected** composé de plusieurs couches linéaires avec :

- activation ReLU
- dropout pour limiter le surapprentissage

La sortie du modèle correspond à la probabilité qu'une déclaration soit vraie.

### 6. Optimisation

Le modèle est entraîné avec :

- la fonction de perte `BCEWithLogitsLoss`
- l'optimiseur `AdamW`

Un **gel initial des poids de BERT** est utilisé au début de l'entraînement afin de stabiliser l'apprentissage du classifieur final.

Après quelques epochs, les poids de BERT sont dégelés afin de permettre un ajustement fin du modèle.

---

# 6. Résultats et Limites de l'approche


J'ai obtenue un résultat de 85,63 pourcents de validation accuracy. Cela est est donc une nette amélioration comparé aux techniques tel que la SVM (validation accuracy ~ 30%) ou même un réseau de neuronne qui n'utilise pas les métadonnées et la crédibilité (validation accuracy ~ 70%).

Bien que l'intégration des métadonnées et de la crédibilité du speaker améliore significativement les performances du modèle, cette approche présente certaines limites.

La principale limite concerne l'utilisation du **score de crédibilité de l'orateur**.

Cette méthode suppose implicitement que le comportement d'un speaker est **relativement stable dans le temps**. En réalité, un individu peut modifier son comportement ou sa stratégie de communication.

Par exemple :

- un orateur historiquement peu fiable pourrait produire soudainement des déclarations correctes
- inversement, un orateur généralement fiable pourrait diffuser des informations erronées dans certaines situations

Dans ces cas, le modèle pourrait être biaisé par l'historique du speaker et attribuer une probabilité incorrecte à la véracité de la déclaration.

On peut également reprocher à cette technique le biais de réputation : Si un locuteur n'a pas d'historique dans la base de données (un "nouveau" speaker), l'efficacité du modèle risque de tomber instantanément. C'est ce qu'on appelle le problème du "Cold Start".

Ces limitations soulignent l'importance de combiner les informations contextuelles avec une **analyse sémantique robuste du texte lui-même**.