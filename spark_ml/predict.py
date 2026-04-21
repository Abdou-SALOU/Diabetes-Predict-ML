"""
=============================================================================
  DiabetesPredict — Module de prédiction
  Module : spark_ml/predict.py
  Description : Fournit les fonctions de chargement du modèle sauvegardé
                et de prédiction pour un nouveau patient.
=============================================================================
"""

from pyspark.ml import PipelineModel


def load_model(spark, model_path="/app/model/best_model"):
    """
    Charge le pipeline Spark ML sauvegardé (VectorAssembler + StandardScaler + Modèle).

    Args:
        spark : SparkSession active
        model_path : Chemin vers le répertoire du modèle sauvegardé

    Returns:
        PipelineModel : Le pipeline complet chargé
    """
    try:
        model = PipelineModel.load(model_path)
        print(f"✅ Modèle chargé depuis : {model_path}")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        raise


def predict(spark, model, input_dict):
    """
    Effectue une prédiction de risque de diabète pour un patient donné.

    Args:
        spark : SparkSession active
        model : PipelineModel chargé
        input_dict : Dictionnaire contenant les 21 features du patient
                     (ex: {"HighBP": 1, "HighChol": 0, "BMI": 28, ...})

    Returns:
        tuple : (prediction, probability)
            - prediction (int) : 0 = pas de diabète, 1 = risque de diabète
            - probability (float) : Probabilité de diabète (entre 0 et 1)
    """
    try:
        # Création du DataFrame Spark à partir du dictionnaire d'entrée
        input_df = spark.createDataFrame([input_dict])

        # Application du pipeline complet (assemblage + scaling + prédiction)
        result = model.transform(input_df)

        # Extraction de la prédiction et de la probabilité
        prediction_row = result.select("prediction", "probability").collect()[0]

        prediction = int(prediction_row["prediction"])
        probability = float(prediction_row["probability"][1])

        return prediction, probability

    except Exception as e:
        print(f"❌ Erreur lors de la prédiction : {e}")
        raise
