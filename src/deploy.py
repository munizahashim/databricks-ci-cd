# Databricks notebook source
# MAGIC %pip install mlflow==3.18.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
model_full_name = dbutils.widgets.get("model_name")

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score
from model_checks import ACCURACY_THRESHOLD, should_promote

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# Read test set written by train task
test_df = spark.read.table(f"{catalog_name}.{schema_name}.wine_test").toPandas()
X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]


def accuracy_of(alias: str) -> float:
    model = mlflow.pyfunc.load_model(f"models:/{model_full_name}@{alias}")
    return accuracy_score(y_test, model.predict(X_test))

# COMMAND ----------

# Score the Challenger and the current Champion on the same held-out data
challenger_version = client.get_model_version_by_alias(model_full_name, "Challenger").version
challenger_accuracy = accuracy_of("Challenger")

try:
    champion_version = client.get_model_version_by_alias(model_full_name, "Champion").version
    champion_accuracy = accuracy_of("Champion")
except MlflowException:
    # First deployment: no Champion yet, so the Challenger is compared to the threshold
    champion_version, champion_accuracy = None, None

with mlflow.start_run():
    mlflow.log_metric("challenger_accuracy", challenger_accuracy)
    if champion_accuracy is not None:
        mlflow.log_metric("champion_accuracy", champion_accuracy)

# COMMAND ----------

if should_promote(challenger_accuracy, champion_accuracy):
    client.set_registered_model_alias(model_full_name, "Champion", challenger_version)
    print(f"Challenger version {challenger_version} ({challenger_accuracy:.4f}) promoted to @Champion.")
elif champion_accuracy is None:
    # No Champion to fall back on, so inference has nothing to load
    raise ValueError(
        f"Challenger accuracy {challenger_accuracy:.4f} is below the required threshold "
        f"{ACCURACY_THRESHOLD:.4f} — pipeline halted. Review the model or lower the threshold if appropriate."
    )
else:
    print(
        f"Challenger version {challenger_version} ({challenger_accuracy:.4f}) did not match "
        f"Champion version {champion_version} ({champion_accuracy:.4f}). @Champion unchanged."
    )
