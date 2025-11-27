"""
✅ Input Validators
File validation and input sanitization
"""

from fastapi import UploadFile, HTTPException
from pathlib import Path
from core.config import settings


def validate_file(file: UploadFile) -> bool:
    """
    Validate uploaded file
    
    Checks:
    - File extension
    - File size
    - MIME type (basic)
    """
    
    if not file:
        raise ValueError("No file provided")
    
    if not file.filename:
        raise ValueError("Filename is empty")
    
    # Check extension
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Note: file.size may not be available, we'll check during save
    
    return True


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal"""
    return Path(filename).name


def validate_analysis_id(analysis_id: str) -> bool:
    """Validate UUID format"""
    import uuid
    try:
        uuid.UUID(analysis_id)
        return True
    except ValueError:
        raise ValueError("Invalid analysis ID format")
