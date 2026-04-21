"""
=============================================================================
  DiabetesPredict — Script d'entraînement Spark ML
  Module : spark_ml/train.py
  Description : Entraîne 3 modèles de classification (Logistic Regression,
                Random Forest, GBT) sur le dataset CDC Diabetes Health
                Indicators, évalue les performances, et sauvegarde le
                meilleur modèle sous forme de Pipeline Spark ML.
=============================================================================
"""

import json
import os
import time

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline


# =============================================================================
# 1. INITIALISATION DE LA SESSION SPARK
# =============================================================================
print("=" * 70)
print("  🚀 DiabetesPredict — Début de l'entraînement Spark ML")
print("=" * 70)

spark = SparkSession.builder \
    .appName("DiabetesPredict-Training") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# Réduire la verbosité des logs Spark
spark.sparkContext.setLogLevel("WARN")

start_time = time.time()

# =============================================================================
# 2. CHARGEMENT DES DONNÉES
# =============================================================================
print("\n📂 Chargement du dataset depuis UCI Machine Learning Repository...")

MODEL_DIR = "/app/model"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

from ucimlrepo import fetch_ucirepo
import pandas as pd

try:
    print("   → Téléchargement via ucimlrepo (ID=891)...")
    cdc_diabetes = fetch_ucirepo(id=891)
    
    # Création du DataFrame Pandas
    X = cdc_diabetes.data.features
    y = cdc_diabetes.data.targets
    
    pdf = pd.concat([X, y], axis=1)
    
    # Conversion du DataFrame Pandas en DataFrame Spark
    print("   → Conversion en DataFrame Spark...")
    df = spark.createDataFrame(pdf)
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors du téléchargement du dataset UCI : {e}")

# Affichage des informations de base
print(f"\n✅ Dataset chargé avec succès !")
print(f"   → Nombre de lignes   : {df.count():,}")
print(f"   → Nombre de colonnes : {len(df.columns)}")

print("\n📊 Schéma du dataset :")
df.printSchema()

print("\n📈 Statistiques descriptives :")
df.describe().show()

# Distribution de la variable cible
print("\n🎯 Distribution de la variable cible (Diabetes_binary) :")
df.groupBy("Diabetes_binary").count().show()

# =============================================================================
# 3. PREPROCESSING
# =============================================================================
print("\n⚙️  Preprocessing des données...")

# Définition des features (toutes les colonnes sauf la cible et éventuel ID)
exclude_cols = ["Diabetes_binary"]
# Vérifier si une colonne ID existe et l'exclure
if "ID" in df.columns:
    exclude_cols.append("ID")

feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"   → Features utilisées ({len(feature_cols)}) : {feature_cols}")

# Assemblage des features en un vecteur unique
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

# Normalisation avec StandardScaler
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=True
)

# Application du preprocessing
print("   → VectorAssembler : assemblage des features...")
print("   → StandardScaler  : normalisation des features...")

# =============================================================================
# 4. SPLIT TRAIN / TEST (80% / 20%)
# =============================================================================
print("\n✂️  Séparation Train/Test (80/20)...")

# On applique d'abord l'assemblage pour faire le split
assembled_df = assembler.transform(df)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

train_data, test_data = scaled_df.randomSplit([0.8, 0.2], seed=42)

print(f"   → Données d'entraînement : {train_data.count():,} lignes")
print(f"   → Données de test         : {test_data.count():,} lignes")

# =============================================================================
# 5. ENTRAÎNEMENT DE 3 MODÈLES (SANS CROSS-VALIDATION POUR DOCKER LOCAL)
# =============================================================================
print("\n" + "=" * 70)
print("  🤖 Entraînement de 3 modèles (Simple Fit pour éviter l'OOM Docker)")
print("=" * 70)

# Évaluateurs
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="Diabetes_binary",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

multiclass_evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="Diabetes_binary",
    predictionCol="prediction",
    metricName="accuracy"
)

multiclass_evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="Diabetes_binary",
    predictionCol="prediction",
    metricName="f1"
)

# Dictionnaire pour stocker les résultats
results = {}

# ─────────────────────────────────────────────────────────────────────────────
# BOUCLE D'ENTRAÎNEMENT DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
models_to_train = {
    "Logistic Regression": LogisticRegression(featuresCol="scaled_features", labelCol="Diabetes_binary", maxIter=100),
    "Random Forest": RandomForestClassifier(featuresCol="scaled_features", labelCol="Diabetes_binary", numTrees=30, maxDepth=4, seed=42),
    "Gradient Boosted Trees": GBTClassifier(featuresCol="scaled_features", labelCol="Diabetes_binary", maxIter=25, maxDepth=5, seed=42)
}

# Mise en cache forcée pour accélérer et stabiliser l'évaluation
train_data.cache()
test_data.cache()

for i, (name, model_estimator) in enumerate(models_to_train.items(), 1):
    print(f"\n📌 [{i}/3] Entraînement — {name}...")
    t_start = time.time()

    fitted_model = model_estimator.fit(train_data)
    predictions = fitted_model.transform(test_data)

    acc = multiclass_evaluator_acc.evaluate(predictions)
    auc = binary_evaluator.evaluate(predictions)
    f1 = multiclass_evaluator_f1.evaluate(predictions)
    t_elapsed = time.time() - t_start

    results[name] = {
        "accuracy": round(acc, 4),
        "auc_roc": round(auc, 4),
        "f1_score": round(f1, 4),
        "training_time_sec": round(t_elapsed, 1),
        "model": fitted_model
    }

    print(f"   ✅ {name} terminé en {t_elapsed:.1f}s")
    print(f"      → Accuracy : {acc:.4f} | AUC-ROC : {auc:.4f} | F1-Score : {f1:.4f}")

# =============================================================================
# 6. TABLEAU COMPARATIF DES MODÈLES
# =============================================================================
print("\n" + "=" * 70)
print("  📊 Tableau comparatif des modèles")
print("=" * 70)
print(f"\n{'Modèle':<28} {'Accuracy':>10} {'AUC-ROC':>10} {'F1-Score':>10} {'Temps (s)':>12}")
print("─" * 72)

for name, metrics in results.items():
    print(
        f"{name:<28} {metrics['accuracy']:>10.4f} {metrics['auc_roc']:>10.4f} "
        f"{metrics['f1_score']:>10.4f} {metrics['training_time_sec']:>12.1f}"
    )

# =============================================================================
# 7. SÉLECTION ET SAUVEGARDE DU MEILLEUR MODÈLE
# =============================================================================
print("\n" + "=" * 70)
print("  💾 Sauvegarde du meilleur modèle")
print("=" * 70)

# Sélection du meilleur modèle basé sur l'AUC-ROC
best_model_name = max(results, key=lambda x: results[x]["auc_roc"])
best_metrics = results[best_model_name]

print(f"\n🏆 Meilleur modèle : {best_model_name}")
print(f"   → AUC-ROC  : {best_metrics['auc_roc']:.4f}")
print(f"   → Accuracy : {best_metrics['accuracy']:.4f}")
print(f"   → F1-Score : {best_metrics['f1_score']:.4f}")

# Création du Pipeline complet pour la sauvegarde
# (VectorAssembler + StandardScaler + Meilleur Modèle)
best_model_obj = best_metrics["model"]

full_pipeline = Pipeline(stages=[assembler, scaler, best_model_obj])
# On ajuste le pipeline sur le dataframe complet (sans le preprocessing déjà appliqué)
full_pipeline_model = full_pipeline.fit(df)

# Nettoyage du répertoire modèle précédent si existant
import shutil
if os.path.exists(BEST_MODEL_PATH):
    shutil.rmtree(BEST_MODEL_PATH)

# Sauvegarde du pipeline complet
full_pipeline_model.save(BEST_MODEL_PATH)
print(f"\n✅ Pipeline complet sauvegardé dans : {BEST_MODEL_PATH}")

# Extraction des feature importances si disponible (RF ou GBT)
feature_importances = None
if best_model_name in ["Random Forest", "Gradient Boosted Trees"]:
    try:
        importances = best_model_obj.featureImportances.toArray()
        feature_importances = {
            feature_cols[i]: round(float(importances[i]), 6)
            for i in range(len(feature_cols))
        }
        # Trier par importance décroissante
        feature_importances = dict(
            sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        )
        print("\n📊 Feature Importances (top 10) :")
        for i, (feat, imp) in enumerate(feature_importances.items()):
            if i >= 10:
                break
            print(f"   {i+1:>2}. {feat:<25} : {imp:.6f}")
    except Exception as e:
        print(f"   ⚠️ Impossible d'extraire les feature importances : {e}")

# Sauvegarde des métriques dans un fichier JSON
metrics_to_save = {
    "best_model": best_model_name,
    "best_metrics": {
        "accuracy": best_metrics["accuracy"],
        "auc_roc": best_metrics["auc_roc"],
        "f1_score": best_metrics["f1_score"],
        "training_time_sec": best_metrics["training_time_sec"],
    },
    "all_models": {
        name: {
            "accuracy": m["accuracy"],
            "auc_roc": m["auc_roc"],
            "f1_score": m["f1_score"],
            "training_time_sec": m["training_time_sec"],
        }
        for name, m in results.items()
    },
    "feature_importances": feature_importances,
    "feature_columns": feature_cols,
    "dataset_info": {
        "total_rows": df.count(),
        "total_features": len(feature_cols),
        "train_rows": train_data.count(),
        "test_rows": test_data.count(),
    },
}

# Créer le répertoire model si nécessaire
os.makedirs(MODEL_DIR, exist_ok=True)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)

print(f"\n✅ Métriques sauvegardées dans : {METRICS_PATH}")

# =============================================================================
# 8. FIN DE L'ENTRAÎNEMENT
# =============================================================================
total_time = time.time() - start_time
print("\n" + "=" * 70)
print(f"  🎉 Entraînement terminé avec succès en {total_time:.1f} secondes !")
print(f"  🏆 Meilleur modèle : {best_model_name} (AUC-ROC = {best_metrics['auc_roc']:.4f})")
print("=" * 70)

# Arrêt propre de la session Spark
spark.stop()
print("\n🔌 Session Spark arrêtée.")
