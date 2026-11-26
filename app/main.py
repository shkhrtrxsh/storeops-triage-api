from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status

from app.classifier import ClassifierError, ClassifierTimeoutError, IncidentClassifier, build_default_classifier
from app.db import DEFAULT_DB_PATH, get_incident_by_id, init_db, list_incidents, save_incident
from app.schemas import IncidentCreate, IncidentTriageResponse


def get_db_path() -> Path:
    return DEFAULT_DB_PATH


def get_classifier() -> IncidentClassifier:
    return build_default_classifier()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db(get_db_path())
    yield


app = FastAPI(
    title="StoreOps Triage API",
    version="0.1.0",
    description="Retail store operations incident triage API with structured LLM classification, ambiguity routing, and SQLite persistence.",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/incidents/triage",
    response_model=IncidentTriageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def triage_incident(
    payload: IncidentCreate,
    classifier: IncidentClassifier = Depends(get_classifier),
    db_path: Path = Depends(get_db_path),
) -> IncidentTriageResponse:
    try:
        classification = await classifier.classify(payload)
    except ClassifierTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except ClassifierError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return save_incident(payload=payload, classification=classification, db_path=db_path)


@app.get("/incidents/{incident_id}", response_model=IncidentTriageResponse)
def read_incident(
    incident_id: int,
    db_path: Path = Depends(get_db_path),
) -> IncidentTriageResponse:
    incident = get_incident_by_id(incident_id=incident_id, db_path=db_path)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )
    return incident


@app.get("/incidents", response_model=list[IncidentTriageResponse])
def read_incidents(
    needs_review: Optional[bool] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db_path: Path = Depends(get_db_path),
) -> list[IncidentTriageResponse]:
    return list_incidents(needs_review=needs_review, limit=limit, db_path=db_path)
