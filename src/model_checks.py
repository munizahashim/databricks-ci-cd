"""
Decision logic for the validate and deploy tasks.

Pure functions only: no Spark, no MLflow calls, no side effects. The notebooks
gather the inputs (input example, tags, accuracy scores) and call these
functions, so CI can unit-test every promotion decision without a workspace.
"""

REQUIRED_TAGS = ("dataset",)
ACCURACY_THRESHOLD = 0.85


def validation_failures(
    has_input_example: bool,
    tags: dict[str, str],
    required_tags: tuple[str, ...] = REQUIRED_TAGS,
) -> list[str]:
    """Return the failed pre-deployment checks. An empty list means the model passed.

    No signature check: Unity Catalog rejects any model version without a
    signature at registration, so every version that reaches this task has one.
    """
    failures = []
    if not has_input_example:
        failures.append("Model has no input example")
    for tag in required_tags:
        if tag not in tags:
            failures.append(f"Required tag '{tag}' is missing from the model version")
    return failures


def should_promote(
    challenger_accuracy: float,
    champion_accuracy: float | None,
    threshold: float = ACCURACY_THRESHOLD,
) -> bool:
    """Decide whether @Challenger replaces @Champion.

    With a Champion in place, the Challenger must perform at least as well as it.
    With no Champion yet (first deployment), the Challenger must meet the threshold.
    """
    if champion_accuracy is None:
        return challenger_accuracy >= threshold
    return challenger_accuracy >= champion_accuracy
