"""
📈 Analysis Result Models
Complete analysis output structure
"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime
from models.slide import Slide, SlideFeatures, SlideScore


class PresentationMetadata(BaseModel):
    """Basic presentation information"""
    filename: str
    file_type: str  # pptx or pdf
    total_slides: int
    analysis_id: str
    uploaded_at: datetime
    analyzed_at: Optional[datetime] = None


class OverallScore(BaseModel):
    """Overall presentation score breakdown"""
    final_score: float  # 0-100
    
    # Component averages
    avg_clarity: float = 0.0
    avg_structure: float = 0.0
    avg_depth: float = 0.0
    avg_design: float = 0.0
    avg_readability: float = 0.0
    avg_coherence: float = 0.0
    
    # Aggregate metrics
    total_redundancy_penalty: float = 0.0
    avg_blooms_score: float = 0.0
    
    # Distribution
    bloom_distribution: Dict[str, int] = {}
    
    # Grade
    letter_grade: str = "F"


class Strengths(BaseModel):
    """Identified strengths"""
    items: List[str] = []


class Weaknesses(BaseModel):
    """Identified weaknesses"""
    items: List[str] = []


class MissingSections(BaseModel):
    """Missing presentation sections"""
    sections: List[str] = []
    suggestions: List[str] = []


class AnalysisResult(BaseModel):
    """Complete analysis result"""
    metadata: PresentationMetadata
    slides: List[Slide]
    features: List[SlideFeatures]
    scores: List[SlideScore]
    overall_score: OverallScore
    strengths: Strengths
    weaknesses: Weaknesses
    missing_sections: MissingSections
    
    # Status
    status: str = "completed"  # pending, processing, completed, failed
    progress: int = 100  # 0-100
    error: Optional[str] = None


class AnalysisStatus(BaseModel):
    """Analysis status for tracking"""
    analysis_id: str
    status: str
    progress: int
    message: Optional[str] = None
