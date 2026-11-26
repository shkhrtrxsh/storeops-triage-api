from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.classifier import ClassifierError, ClassifierTimeoutError, RuleBasedClassifier
from app.db import init_db, list_incidents
from app.main import app, get_classifier, get_db_path
from app.schemas import IncidentCreate


@pytest.fixture()
def client(tmp_path: Path):
    db_file = tmp_path / "test_storeops.db"
    init_db(db_file)

    app.dependency_overrides[get_db_path] = lambda: db_file
    app.dependency_overrides[get_classifier] = lambda: RuleBasedClassifier()

    with TestClient(app) as test_client:
        yield test_client, db_file

    app.dependency_overrides.clear()


def test_health_check(client):
    test_client, _ = client
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_triage_clear_pos_incident_and_retrieve(client):
    test_client, _ = client
    payload = {
        "store_id": "DOM-DEL-1042",
        "title": "POS billing terminals down",
        "description": "All 3 POS counters stopped accepting UPI payments during dinner rush.",
    }
    create_resp = test_client.post("/incidents/triage", json=payload)
    assert create_resp.status_code == 201
    body = create_resp.json()
    assert body["suggested_category"] == "pos_billing"
    assert body["suggested_priority"] == "P1"
    assert body["needs_review"] is False

    get_resp = test_client.get(f"/incidents/{body['id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["store_id"] == "DOM-DEL-1042"
    assert get_resp.json()["suggested_category"] == "pos_billing"


def test_triage_ambiguous_incident_routes_to_human_review(client):
    test_client, _ = client
    payload = {
        "store_id": "DOM-BLR-0211",
        "title": "Something is wrong in store",
        "description": "Not working properly since morning, please check asap.",
    }
    resp = test_client.post("/incidents/triage", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["needs_review"] is True
    assert body["suggested_category"] is None
    assert body["suggested_priority"] is None


def test_triage_conflicting_subsystems_routes_to_human_review(client):
    test_client, _ = client
    payload = {
        "store_id": "DOM-MUM-0881",
        "title": "POS and printer and internet issue",
        "description": "POS billing failed while receipt printer jammed and store Wi-Fi dropped.",
    }
    resp = test_client.post("/incidents/triage", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["needs_review"] is True
    assert body["suggested_category"] is None


def test_invalid_input_returns_422(client):
    test_client, _ = client
    resp = test_client.post(
        "/incidents/triage",
        json={"store_id": " ", "title": "bad", "description": "short"},
    )
    assert resp.status_code == 422


def test_unknown_incident_id_returns_404(client):
    test_client, _ = client
    resp = test_client.get("/incidents/999999")
    assert resp.status_code == 404


def test_classifier_timeout_does_not_persist_incident(client):
    test_client, db_file = client

    class TimeoutClassifier:
        async def classify(self, incident: IncidentCreate):
            raise ClassifierTimeoutError("Upstream LLM timed out")

    app.dependency_overrides[get_classifier] = lambda: TimeoutClassifier()

    resp = test_client.post(
        "/incidents/triage",
        json={
            "store_id": "DOM-NOI-0012",
            "title": "KDS screen frozen",
            "description": "Kitchen display screen stopped updating incoming orders.",
        },
    )
    assert resp.status_code == 504
    assert list_incidents(db_path=db_file) == []


def test_classifier_malformed_output_does_not_persist_incident(client):
    test_client, db_file = client

    class BrokenClassifier:
        async def classify(self, incident: IncidentCreate):
            raise ClassifierError("Malformed JSON from model")

    app.dependency_overrides[get_classifier] = lambda: BrokenClassifier()

    resp = test_client.post(
        "/incidents/triage",
        json={
            "store_id": "DOM-NOI-0012",
            "title": "KDS screen frozen",
            "description": "Kitchen display screen stopped updating incoming orders.",
        },
    )
    assert resp.status_code == 502
    assert list_incidents(db_path=db_file) == []


def test_list_incidents_filter_needs_review(client):
    test_client, _ = client
    test_client.post(
        "/incidents/triage",
        json={
            "store_id": "DOM-DEL-1001",
            "title": "Store internet outage",
            "description": "Primary ISP link is completely down across the entire store.",
        },
    )
    test_client.post(
        "/incidents/triage",
        json={
            "store_id": "DOM-DEL-1002",
            "title": "Weird issue reported",
            "description": "Something is wrong here, please check.",
        },
    )

    review_only = test_client.get("/incidents?needs_review=true")
    assert review_only.status_code == 200
    items = review_only.json()
    assert len(items) == 1
    assert items[0]["store_id"] == "DOM-DEL-1002"
