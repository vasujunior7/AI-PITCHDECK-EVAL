"""
📊 Slide Data Models
Structured representation of presentation slides
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class BloomLevel(str, Enum):
    """Bloom's Taxonomy Levels"""
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class TextBox(BaseModel):
    """Text box with coordinates"""
    text: str
    x: float
    y: float
    width: float
    height: float


class ImageInfo(BaseModel):
    """Image metadata"""
    path: str
    width: int
    height: int
    format: Optional[str] = None


class Slide(BaseModel):
    """Complete slide representation"""
    slide_number: int
    title: Optional[str] = None
    body_text: str = ""
    notes: Optional[str] = None
    text_boxes: List[TextBox] = []
    images: List[ImageInfo] = []
    image_count: int = 0
    
    # Layout metrics
    total_words: int = 0
    layout_density: float = 0.0
    
    # Section classification (Milestone 4)
    section: Optional[str] = None
    section_confidence: float = 0.0
    
    # Computed features (filled during analysis)
    section: Optional[str] = None
    section_confidence: Optional[float] = None
    
    class Config:
        use_enum_values = True


class SlideFeatures(BaseModel):
    """Advanced features extracted from slide"""
    slide_number: int
    
    # Semantic features
    semantic_density: float = 0.0
    embedding: Optional[List[float]] = None
    
    # Redundancy
    redundancy_score: float = 0.0
    is_redundant: bool = False
    
    # Image-text alignment
    image_text_alignment: float = 0.0
    
    # Layout quality
    is_crowded: bool = False
    is_sparse: bool = False
    layout_quality_score: float = 0.0
    
    # Coherence
    coherence_score: float = 0.0
    
    # Text quality
    readability_score: float = 0.0
    lexical_richness: float = 0.0
    keywords: List[str] = []
    
    # Bloom's taxonomy
    bloom_level: Optional[BloomLevel] = None
    bloom_confidence: float = 0.0
    
    class Config:
        use_enum_values = True


class SlideScore(BaseModel):
    """Scoring for individual slide"""
    slide_number: int
    
    # Component scores (0-100)
    clarity_score: float = 0.0
    structure_score: float = 0.0
    depth_score: float = 0.0
    design_score: float = 0.0
    readability_score: float = 0.0
    coherence_score: float = 0.0
    
    # Penalties
    redundancy_penalty: float = 0.0
    
    # Bloom's score
    blooms_score: float = 0.0
    
    # Final weighted score
    final_score: float = 0.0
    
    # AI feedback
    ai_feedback: Optional[Dict[str, Any]] = None
