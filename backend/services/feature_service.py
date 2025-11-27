"""
🧠 Feature Service - PLACEHOLDER
Will be implemented in Milestone 5
"""

from typing import List
from models.slide import Slide, SlideFeatures


def extract_features(slides: List[Slide]) -> List[SlideFeatures]:
    """
    Extract advanced features
    
    TODO: Implement in Milestone 5
    """
    return [
        SlideFeatures(slide_number=slide.slide_number)
        for slide in slides
    ]
