from pydantic import BaseModel, Field

class ReportRequest(BaseModel):
    report_text: str = Field(..., min_length=20)

class ReportResponse(BaseModel):
    predicted_category: str
    confidence: float
