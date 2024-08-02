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


@pytest.fixture
def body_wrong():
    return {
        "latitude": 91,
        "longitude": -181,
        "capacity_kwp": -1,
        "tilt": 91,
        "orientation": 361,
        "inverter_type": "string",
    }


def test_api_ok(client, body):
    response = client.post("/forecast/", json=body)
    response_body = response.json()
    assert response.status_code == 200
    assert isinstance(response_body["power_kw"], dict)
    assert any("power_kw" in key for key in response.json().keys()), f"No key contains power_kw"
    assert len(response.json()["power_kw"]) == 192


def test_api_ok_short(client, body_short):
    response = client.post("/forecast/", json=body_short)
    response_body = response.json()

    assert response.status_code == 200
    assert isinstance(response_body["power_kw"], dict)
    assert any("power_kw" in key for key in response_body.keys()), f"No key contains power_kw"
    assert len(response.json()["power_kw"]) == 192


def test_api_wrong_body(client, body_wrong):
    # Tests all wrong data types responses
    response = client.post("/forecast/", json=body_wrong)
    response_body = response.json()

    expected_response = {
        "detail": [
            {
                "loc": ["body", "latitude"],
                "msg": "Input should be less than or equal to 90",
                "type": "less_than_equal",
                "input": 91,
                "ctx": {"le": 90.0},
            },
            {
                "loc": ["body", "longitude"],
                "msg": "Input should be greater than or equal to -180",
                "type": "greater_than_equal",
                "input": -181,
                "ctx": {"ge": -180.0},
            },
            {
                "loc": ["body", "capacity_kwp"],
                "msg": "Input should be greater than or equal to 0",
                "type": "greater_than_equal",
                "input": -1,
                "ctx": {"ge": 0.0},
            },
            {
                "loc": ["body", "tilt"],
                "msg": "Input should be less than or equal to 90",
                "type": "less_than_equal",
                "input": 91,
                "ctx": {"le": 90.0},
            },
            {
                "loc": ["body", "orientation"],
                "msg": "Input should be less than or equal to 360",
                "type": "less_than_equal",
                "input": 361,
                "ctx": {"le": 360.0},
            },
        ]
    }
    assert response.status_code == 422
    assert response_body == expected_response
