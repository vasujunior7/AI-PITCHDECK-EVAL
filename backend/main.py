"""
🚀 AI-Powered Presentation Analyzer - Main Entry Point
Production-grade FastAPI backend with middleware & routing
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import uuid
from pathlib import Path

from routers import upload, analyze, score, improve, complete_analysis
from middleware.request_id import RequestIDMiddleware
from middleware.error_handler import setup_exception_handlers
from core.config import settings
from core.logging import setup_logging, logger

# Setup logging
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events - startup & shutdown"""
    logger.info("Starting AI Presentation Analyzer...")
    
    # Create necessary directories
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"Reports directory: {settings.REPORTS_DIR}")
    logger.info("Server ready!")
    
    yield
    
    logger.info("Shutting down gracefully...")

# Initialize FastAPI app
app = FastAPI(
    title="AI Presentation Analyzer",
    description="Enterprise-grade PPT/PDF quality assessment with AI insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Middleware
app.add_middleware(RequestIDMiddleware)

# Setup Exception Handlers
setup_exception_handlers(app)

# Include Routers
app.include_router(complete_analysis.router, tags=["Complete Analysis"])  # NEW: Full pipeline
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(score.router, prefix="/api", tags=["Scoring"])
app.include_router(improve.router, prefix="/api", tags=["Improvements"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "alive",
        "service": "AI Presentation Analyzer",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "upload_dir_exists": Path(settings.UPLOAD_DIR).exists(),
        "reports_dir_exists": Path(settings.REPORTS_DIR).exists(),
    }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    
    # Get or create request ID
    request_id = request.state.request_id if hasattr(request.state, 'request_id') else str(uuid.uuid4())
    
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"[{request_id}] Completed in {process_time:.2f}s - Status: {response.status_code}")
    
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
