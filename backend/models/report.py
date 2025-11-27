"""
📄 Report Models
Improvement suggestions and recommendations
"""

from pydantic import BaseModel
from typing import List, Optional


class SlideImprovement(BaseModel):
    """Improvement suggestion for a slide"""
    slide_number: int
    
    # Original content
    original_title: Optional[str] = None
    original_text: str
    
    # Improved content
    improved_title: Optional[str] = None
    improved_text: str
    
    # Rationale
    issues_identified: List[str] = []
    improvements_made: List[str] = []
    
    # Design suggestions
    visual_suggestions: List[str] = []
    
    # Knowledge depth upgrade
    original_bloom_level: Optional[str] = None
    target_bloom_level: Optional[str] = None
    bloom_upgrade_suggestion: Optional[str] = None


class GlobalRecommendations(BaseModel):
    """Overall recommendations"""
    quick_wins: List[str] = []
    major_improvements: List[str] = []
    design_recommendations: List[str] = []
    content_recommendations: List[str] = []


class ImprovementReport(BaseModel):
    """Complete improvement report"""
    analysis_id: str
    slide_improvements: List[SlideImprovement]
    global_recommendations: GlobalRecommendations
    executive_summary: str
