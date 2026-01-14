from fastapi import FastAPI
from app.api.routes import router

# Create FastAPI app
app = FastAPI(
    title="Whistleblower AI Service",
    description="AI-powered classification and triaging of whistleblower reports",
    version="0.1.0"
)

# Include API routes
app.include_router(router)

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to Whistleblower AI Service. Use /classify to classify reports."}
