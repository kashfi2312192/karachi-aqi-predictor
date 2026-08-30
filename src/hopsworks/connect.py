import os
from pathlib import Path

from dotenv import load_dotenv
import hopsworks


# ============================================================
# PEARLS AQI PREDICTOR
# HOPSWORKS CONNECTION
# ============================================================

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(
    PROJECT_ROOT / ".env"
)


def get_hopsworks_project():

    project_name = os.getenv(
        "HOPSWORKS_PROJECT"
    )

    api_key = os.getenv(
        "HOPSWORKS_API_KEY"
    )

    host = os.getenv(
        "HOPSWORKS_HOST"
    )

    if not project_name:
        raise RuntimeError(
            "HOPSWORKS_PROJECT is missing from .env"
        )

    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY is missing from .env"
        )

    login_kwargs = {
        "project": project_name,
        "api_key_value": api_key,
        "engine": "python",
    }

    if host:
        login_kwargs["host"] = host

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("HOPSWORKS CONNECTION")
    print("=" * 70)

    print("\nConnecting to Hopsworks...")

    project = hopsworks.login(
        **login_kwargs
    )

    print(
        f"Connected successfully."
    )

    print(
        f"Project: {project.name}"
    )

    return project


def get_feature_store():

    project = get_hopsworks_project()

    feature_store = (
        project
        .get_feature_store()
    )

    print(
        f"Feature Store: "
        f"{feature_store.name}"
    )

    return feature_store


if __name__ == "__main__":

    project = get_hopsworks_project()

    print("\nHopsworks connection test PASSED.")