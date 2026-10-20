"""HTTP client boundary for AdaptiveAI frontend/tests (no Streamlit).

Pure requests-based client for the FastAPI service. Used by Python
integration tests; the React frontend uses its own typed TS client
(`frontend/src/api/client.ts`) with the same endpoints.
"""

import os
from typing import Any, Dict, Optional

import requests

API_URL_ENV_VAR = "ADAPTIVE_AI_API_URL"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
PREDICT_PATH = "/predict/house-price"
MONITOR_PATH = "/monitor/house-price"
MONITOR_HISTORY_PATH = "/monitor/house-price/history"
RETRAIN_PATH = "/monitor/house-price/retrain"
RETRAIN_HISTORY_PATH = "/monitor/house-price/retrain/history"
RETRAIN_STATUS_PATH = "/monitor/house-price/retrain/status"
MODEL_HISTORY_PATH = RETRAIN_HISTORY_PATH
REQUEST_TIMEOUT_SECONDS = 10


class DashboardAPIError(Exception):
    """User-safe API failure (never carries internals)."""


def get_api_base_url() -> str:
    raw = os.environ.get(API_URL_ENV_VAR) or DEFAULT_API_BASE_URL
    return raw.rstrip("/")


def predict_house_price_via_api(
    area_sqft: float,
    bedrooms: int,
    bathrooms: int,
    property_age: float,
    parking: int,
    location: str = "Wakad",
    floor_number: int = 2,
    total_floors: int = 12,
    furnishing_status: str = "Semi-Furnished",
    property_type: str = "Apartment",
    base_url: Optional[str] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> float:
    payload = {
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "location": location,
        "property_age": property_age,
        "floor_number": floor_number,
        "total_floors": total_floors,
        "furnishing_status": furnishing_status,
        "property_type": property_type,
    }
    url = f"{base_url or get_api_base_url()}{PREDICT_PATH}"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise DashboardAPIError(
            "Prediction API is unavailable. Start the FastAPI service and retry."
        ) from e
    if response.status_code != 200:
        raise DashboardAPIError(
            f"Prediction request failed (HTTP {response.status_code}). "
            "Check the inputs or retry later."
        )
    try:
        data = response.json()
    except ValueError as e:
        raise DashboardAPIError(
            "Prediction API returned an invalid response. Retry later."
        ) from e
    prediction = data.get("prediction") if isinstance(data, dict) else None
    if isinstance(prediction, bool) or not isinstance(prediction, (int, float)):
        raise DashboardAPIError(
            "Prediction API returned an invalid response. Retry later."
        )
    return float(prediction)


def run_house_price_drift_check_via_api(
    base_url: Optional[str] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    url = f"{base_url or get_api_base_url()}{MONITOR_PATH}"
    try:
        response = requests.post(url, timeout=timeout)
    except requests.RequestException as e:
        raise DashboardAPIError(
            "Monitoring API is unavailable. Start the FastAPI service and retry."
        ) from e
    if response.status_code != 200:
        raise DashboardAPIError(
            f"Monitoring request failed (HTTP {response.status_code}). "
            "Retry later."
        )
    try:
        data = response.json()
    except ValueError as e:
        raise DashboardAPIError(
            "Drift Check API returned an invalid response. Retry later."
        ) from e
    if not isinstance(data, dict):
        raise DashboardAPIError(
            "Drift Check API returned an invalid response. Retry later."
        )
    return dict(data)


def get_house_price_model_history_via_api(
    limit: int = 50,
    offset: int = 0,
    base_url: Optional[str] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise DashboardAPIError("Model history request failed. Retry later.")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise DashboardAPIError("Model history request failed. Retry later.")
    url = f"{base_url or get_api_base_url()}{RETRAIN_HISTORY_PATH}"
    try:
        response = requests.get(
            url, params={"limit": limit, "offset": offset}, timeout=timeout
        )
    except requests.RequestException as e:
        raise DashboardAPIError(
            "Model history API is unavailable. Start the FastAPI service and retry."
        ) from e
    if response.status_code != 200:
        raise DashboardAPIError(
            f"Model history request failed (HTTP {response.status_code}). "
            "Retry later."
        )
    try:
        data = response.json()
    except ValueError as e:
        raise DashboardAPIError(
            "Model history API returned an invalid response. Retry later."
        ) from e
    if not isinstance(data, dict):
        raise DashboardAPIError(
            "Model history API returned an invalid response. Retry later."
        )
    items = data.get("items")
    if not isinstance(items, list):
        raise DashboardAPIError(
            "Model history API returned an invalid response. Retry later."
        )
    return dict(data)


def get_house_price_monitoring_history_via_api(
    limit: int = 100,
    offset: int = 0,
    base_url: Optional[str] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise DashboardAPIError("Monitoring history request failed. Retry later.")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise DashboardAPIError("Monitoring history request failed. Retry later.")
    url = f"{base_url or get_api_base_url()}{MONITOR_HISTORY_PATH}"
    try:
        response = requests.get(
            url, params={"limit": limit, "offset": offset}, timeout=timeout
        )
    except requests.RequestException as e:
        raise DashboardAPIError(
            "Monitoring API is unavailable. Start the FastAPI service and retry."
        ) from e
    if response.status_code != 200:
        raise DashboardAPIError(
            f"Monitoring history request failed (HTTP {response.status_code}). "
            "Retry later."
        )
    try:
        data = response.json()
    except ValueError as e:
        raise DashboardAPIError(
            "Monitoring API returned an invalid response. Retry later."
        ) from e
    if not isinstance(data, dict):
        raise DashboardAPIError(
            "Monitoring API returned an invalid response. Retry later."
        )
    items = data.get("items")
    if not isinstance(items, list):
        raise DashboardAPIError(
            "Monitoring API returned an invalid response. Retry later."
        )
    return dict(data)


def trigger_house_price_retraining_via_api(
    promote: bool = True,
    base_url: Optional[str] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    if not isinstance(promote, bool):
        raise DashboardAPIError("Retraining request failed. Retry later.")
    url = f"{base_url or get_api_base_url()}{RETRAIN_PATH}"
    try:
        response = requests.post(url, json={"promote": promote}, timeout=timeout)
    except requests.RequestException as e:
        raise DashboardAPIError(
            "Retraining API is unavailable. Start the FastAPI service and retry."
        ) from e
    if response.status_code != 200:
        raise DashboardAPIError(
            f"Retraining request failed (HTTP {response.status_code}). "
            "Retry later."
        )
    try:
        data = response.json()
    except ValueError as e:
        raise DashboardAPIError(
            "Retraining API returned an invalid response. Retry later."
        ) from e
    if not isinstance(data, dict):
        raise DashboardAPIError(
            "Retraining API returned an invalid response. Retry later."
        )
    return dict(data)


def get_house_price_retraining_history_via_api(
    limit: int = 50,
    offset: int = 0,
    base_url: Optional[str] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise DashboardAPIError("Retraining history request failed. Retry later.")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise DashboardAPIError("Retraining history request failed. Retry later.")
    url = f"{base_url or get_api_base_url()}{RETRAIN_HISTORY_PATH}"
    try:
        response = requests.get(
            url, params={"limit": limit, "offset": offset}, timeout=timeout
        )
    except requests.RequestException as e:
        raise DashboardAPIError(
            "Retraining API is unavailable. Start the FastAPI service and retry."
        ) from e
    if response.status_code != 200:
        raise DashboardAPIError(
            f"Retraining history request failed (HTTP {response.status_code}). "
            "Retry later."
        )
    try:
        data = response.json()
    except ValueError as e:
        raise DashboardAPIError(
            "Retraining API returned an invalid response. Retry later."
        ) from e
    if not isinstance(data, dict):
        raise DashboardAPIError(
            "Retraining API returned an invalid response. Retry later."
        )
    items = data.get("items")
    if not isinstance(items, list):
        raise DashboardAPIError(
            "Retraining API returned an invalid response. Retry later."
        )
    return dict(data)


def get_house_price_retraining_status_via_api(
    base_url: Optional[str] = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    url = f"{base_url or get_api_base_url()}{RETRAIN_STATUS_PATH}"
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        raise DashboardAPIError(
            "Retraining API is unavailable. Start the FastAPI service and retry."
        ) from e
    if response.status_code != 200:
        raise DashboardAPIError(
            f"Scheduler status request failed (HTTP {response.status_code}). "
            "Retry later."
        )
    try:
        data = response.json()
    except ValueError as e:
        raise DashboardAPIError(
            "Retraining API returned an invalid response. Retry later."
        ) from e
    if not isinstance(data, dict):
        raise DashboardAPIError(
            "Retraining API returned an invalid response. Retry later."
        )
    return dict(data)
