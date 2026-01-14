from fastapi import APIRouter
from app.schemas.report import ReportRequest, ReportResponse
from app.ml.classifier import classify_report

# Create a router for API endpoints
router = APIRouter()

# Health check endpoint
@router.get("/health")
def health_check():
    return {"status": "ok", "service": "whistleblower-ai"}

# Report classification endpoint
@router.post("/classify", response_model=ReportResponse)
def classify_report_endpoint(report: ReportRequest):
    """
    Receives a whistleblower report and returns predicted category with confidence.
    """
    # Call the ML classifier
    category, confidence = classify_report(report.report_text)
    
    # Return response in Pydantic schema
    return ReportResponse(predicted_category=category, confidence=confidence)
