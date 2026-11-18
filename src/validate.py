# Databricks notebook source
# MAGIC %pip install mlflow==3.18.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

model_full_name = dbutils.widgets.get("model_name")
# Set by train_task; the job passes it in as {{tasks.train_task.values.model_version}}
model_version = dbutils.widgets.get("model_version")

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from model_checks import validation_failures

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# Mark the version as under validation; updated to PASSED or FAILED below
client.set_model_version_tag(model_full_name, model_version, "model_validation_status", "PENDING")

# COMMAND ----------

model_info = mlflow.models.get_model_info(f"models:/{model_full_name}/{model_version}")
version_tags = client.get_model_version(model_full_name, model_version).tags

failures = validation_failures(
    has_input_example=model_info.saved_input_example_info is not None,
    tags=version_tags,
)

# COMMAND ----------

if failures:
    client.set_model_version_tag(model_full_name, model_version, "model_validation_status", "FAILED")
    raise ValueError(
        f"Version {model_version} failed validation: {'; '.join(failures)} — pipeline halted."
    )

client.set_model_version_tag(model_full_name, model_version, "model_validation_status", "PASSED")
client.set_registered_model_alias(model_full_name, "Challenger", model_version)
print(f"Version {model_version} passed validation and is now @Challenger.")
