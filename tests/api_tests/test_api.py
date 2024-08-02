import pytest
from fastapi.testclient import TestClient

from api.app.api import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def body():
    return {
        "latitude": -90,
        "longitude": -180,
        "capacity_kwp": 0,
        "tilt": 35,
        "orientation": 180,
        "inverter_type": "string",
    }


@pytest.fixture
def body_short():
    return {
        "latitude": -90,
        "longitude": -180,
        "capacity_kwp": 0,
    }


def test_api(client, body):
    response = client.post("/forecast/", json=body)

    print(response.json().keys())
    assert response.status_code == 200
    assert any("power_kw" in key for key in response.json().keys()), f"No key contains power_kw"
    assert len(response.json()["power_kw"]) == 192


def test_api_short(client, body_short):
    response = client.post("/forecast/", json=body_short)

    print(response.json().keys())
    assert response.status_code == 200
    assert any("power_kw" in key for key in response.json().keys()), f"No key contains power_kw"
    assert len(response.json()["power_kw"]) == 192
