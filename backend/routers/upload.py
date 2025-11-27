"""
📤 Upload Router
Handles file uploads with validation
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pathlib import Path
import uuid
import aiofiles
from datetime import datetime

from core.config import settings
from core.security import check_rate_limit
from core.logging import logger
from models.analysis import PresentationMetadata
from utils.validators import validate_file

router = APIRouter()


@router.post("/upload", dependencies=[Depends(check_rate_limit)])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a presentation file (PPTX or PDF)
    
    Returns:
        - analysis_id: Unique ID for tracking
        - filename: Original filename
        - file_type: pptx or pdf
        - status: Upload status
    """
    
    # Validate file
    try:
        validate_file(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    
    # Get file extension
    file_extension = Path(file.filename).suffix.lower()
    file_type = file_extension.replace('.', '')
    
    # Create unique filename
    unique_filename = f"{analysis_id}{file_extension}"
    file_path = Path(settings.UPLOAD_DIR) / unique_filename
    
    logger.info(f"📥 Uploading file: {file.filename} → {unique_filename}")
    
    # Save file asynchronously
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        file_size = len(content) / (1024 * 1024)  # MB
        logger.info(f"✅ File saved: {file_path} ({file_size:.2f} MB)")
        
    except Exception as e:
        logger.error(f"❌ Failed to save file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save file")
    
    # Create metadata
    metadata = PresentationMetadata(
        filename=file.filename,
        file_type=file_type,
        total_slides=0,  # Will be updated during parsing
        analysis_id=analysis_id,
        uploaded_at=datetime.now()
    )
    
    return {
        "analysis_id": analysis_id,
        "filename": file.filename,
        "file_type": file_type,
        "file_size_mb": round(file_size, 2),
        "status": "uploaded",
        "message": "File uploaded successfully. Ready for analysis."
    }


@router.get("/upload/{analysis_id}/status")
async def get_upload_status(analysis_id: str):
    """Check if uploaded file exists"""
    
    # Check for both extensions
    for ext in settings.ALLOWED_EXTENSIONS:
        file_path = Path(settings.UPLOAD_DIR) / f"{analysis_id}{ext}"
        if file_path.exists():
            return {
                "analysis_id": analysis_id,
                "status": "found",
                "file_path": str(file_path),
                "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
            }
    
    raise HTTPException(status_code=404, detail="File not found")
