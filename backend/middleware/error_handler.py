"""
⚠️ Global Error Handler
Catches all exceptions and returns structured responses
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from core.logging import logger
import traceback


def setup_exception_handlers(app: FastAPI):
    """Setup global exception handlers"""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors"""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.warning(f"⚠️ [{request_id}] Validation error: {exc.errors()}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": exc.errors(),
                "request_id": request_id
            }
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Catch-all exception handler"""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.error(f"❌ [{request_id}] Unhandled exception: {str(exc)}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": str(exc),
                "request_id": request_id
            }
        )
