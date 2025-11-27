"""
🤖 Evaluation Service - PLACEHOLDER
Will be implemented in Milestone 6
"""

from typing import List
from models.slide import Slide, SlideFeatures, SlideScore


def evaluate_slides(slides: List[Slide], features: List[SlideFeatures]) -> List[SlideScore]:
    """
    AI evaluation with GPT-4
    
    TODO: Implement in Milestone 6
    """
    return [
        SlideScore(slide_number=slide.slide_number, final_score=75.0)
        for slide in slides
    ]
