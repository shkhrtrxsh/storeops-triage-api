from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class IncidentCategory(str, Enum):
    POS_BILLING = "pos_billing"
    KITCHEN_DISPLAY = "kitchen_display"
    NETWORK_CONNECTIVITY = "network_connectivity"
    INVENTORY_SYNC = "inventory_sync"
    HARDWARE_PERIPHERAL = "hardware_peripheral"


class IncidentPriority(str, Enum):
    P1_CRITICAL = "P1"
    P2_HIGH = "P2"
    P3_MEDIUM = "P3"
    P4_LOW = "P4"


class IncidentCreate(BaseModel):
    store_id: str = Field(..., min_length=2, max_length=32, examples=["DOM-DEL-1042"])
    title: str = Field(..., min_length=5, max_length=160, examples=["POS terminal frozen during peak dinner rush"])
    description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        examples=["All 3 billing counters at store DOM-DEL-1042 stopped accepting UPI payments after 8:15 PM."],
    )

    @field_validator("store_id", "title", "description")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be blank or whitespace only")
        return cleaned


class ClassificationResult(BaseModel):
    suggested_category: Optional[IncidentCategory] = None
    suggested_priority: Optional[IncidentPriority] = None
    needs_review: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., min_length=3, max_length=500)


class IncidentTriageResponse(BaseModel):
    id: int
    store_id: str
    title: str
    description: str
    suggested_category: Optional[IncidentCategory] = None
    suggested_priority: Optional[IncidentPriority] = None
    needs_review: bool
    confidence: float
    reason: str
    created_at: datetime
