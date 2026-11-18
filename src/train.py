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
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="label")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

with mlflow.start_run():
    n_estimators = 100
    max_depth = 5

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("dataset", "sklearn_wine")

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="weighted"))

    signature = infer_signature(X_test, y_pred)
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=X_test[:5],
    )

# COMMAND ----------

# Register the logged model as a new version in Unity Catalog.
model_version = mlflow.register_model(model_info.model_uri, model_full_name).version

# Record the training dataset on the version. validate_task requires this tag.
client = MlflowClient()
client.set_model_version_tag(model_full_name, model_version, "dataset", "sklearn_wine")

# Pass the version number to validate_task. No alias yet:
# a version earns @Challenger only after it passes validation.
dbutils.jobs.taskValues.set(key="model_version", value=model_version)
print(f"Registered {model_full_name} version {model_version}")

# COMMAND ----------

# Write test set to Delta for the deploy and inference tasks to consume
test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
spark.createDataFrame(test_df).write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog_name}.{schema_name}.wine_test"
)
# Wine quality classifier — RandomForest baseline
