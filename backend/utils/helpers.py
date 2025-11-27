"""
🛠️ Helper Utilities
General helper functions
"""

from typing import List, Dict, Any
import re


def calculate_letter_grade(score: float) -> str:
    """Convert numeric score to letter grade"""
    
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def clean_text(text: str) -> str:
    """Basic text cleaning"""
    
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.,!?;:\-\(\)]', '', text)
    
    return text.strip()


def count_words(text: str) -> int:
    """Count words in text"""
    if not text:
        return 0
    return len(text.split())


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def calculate_average(values: List[float]) -> float:
    """Calculate average of list"""
    if not values:
        return 0.0
    return sum(values) / len(values)


def normalize_score(score: float, min_val: float = 0, max_val: float = 100) -> float:
    """Normalize score to 0-100 range"""
    if score < min_val:
        return 0.0
    if score > max_val:
        return 100.0
    return float(score)
