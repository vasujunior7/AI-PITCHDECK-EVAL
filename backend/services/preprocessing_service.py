"""
🧹 Advanced Text Preprocessing Service
Cleans, analyzes, and enriches slide text with quality metrics

Features:
- Text cleaning and normalization
- Readability scoring (Flesch-Kincaid, Gunning Fog)
- Lexical richness (Type-Token Ratio)
- Keyword extraction (YAKE)
- Filler word detection
"""

from typing import List, Dict, Tuple
import re
import string

# Readability metrics
import textstat

# Keyword extraction
import yake

# NLP
import nltk
from nltk.corpus import stopwords

from models.slide import Slide
from core.logging import logger

# Download NLTK data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Common filler words in presentations
FILLER_WORDS = {
    'um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally',
    'just', 'really', 'very', 'quite', 'somewhat', 'sort of', 'kind of',
    'i mean', 'i think', 'i guess', 'maybe', 'perhaps'
}

# Boilerplate phrases to remove
BOILERPLATE_PATTERNS = [
    r'slide \d+',
    r'page \d+',
    r'confidential',
    r'proprietary',
    r'all rights reserved',
]


def preprocess_slides(slides: List[Slide]) -> List[Slide]:
    """
    Preprocess all slides with text analysis
    
    Enriches each slide with:
    - Cleaned text
    - Readability scores
    - Lexical richness
    - Keywords
    """
    
    logger.info(f"Preprocessing {len(slides)} slides...")
    
    for i, slide in enumerate(slides, 1):
        try:
            # Preprocess text
            enriched_data = preprocess_text(slide.body_text)
            
            # Update slide (we'll store this in slide attributes or return separately)
            # For now, we keep original text but log the metrics
            logger.info(
                f"Slide {slide.slide_number}: "
                f"Readability={enriched_data['readability_score']:.1f}, "
                f"Richness={enriched_data['richness_score']:.2f}, "
                f"Keywords={len(enriched_data['keywords'])}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to preprocess slide {slide.slide_number}: {str(e)}")
    
    logger.info("Preprocessing complete")
    
    return slides


def preprocess_text(text: str) -> Dict:
    """
    Comprehensive text preprocessing and analysis
    
    Returns enriched data dictionary with:
    - clean_text: Cleaned and normalized text
    - readability_score: Flesch Reading Ease (0-100, higher = easier)
    - readability_grade: Flesch-Kincaid Grade Level
    - gunning_fog: Gunning Fog Index
    - richness_score: Type-Token Ratio (lexical diversity)
    - keywords: List of key terms
    - filler_word_count: Number of filler words detected
    """
    
    if not text or len(text.strip()) < 3:
        return {
            'clean_text': text,
            'readability_score': 0.0,
            'readability_grade': 0.0,
            'gunning_fog': 0.0,
            'richness_score': 0.0,
            'keywords': [],
            'filler_word_count': 0
        }
    
    # Step 1: Clean text
    clean_text = clean_presentation_text(text)
    
    # Step 2: Calculate readability scores
    readability_score = calculate_readability(clean_text)
    
    # Step 3: Calculate lexical richness
    richness_score = calculate_lexical_richness(clean_text)
    
    # Step 4: Extract keywords
    keywords = extract_keywords(clean_text, max_keywords=5)
    
    # Step 5: Detect filler words
    filler_count = count_filler_words(text)
    
    return {
        'clean_text': clean_text,
        'readability_score': readability_score['flesch_reading_ease'],
        'readability_grade': readability_score['flesch_kincaid_grade'],
        'gunning_fog': readability_score['gunning_fog'],
        'richness_score': richness_score,
        'keywords': keywords,
        'filler_word_count': filler_count
    }


def clean_presentation_text(text: str) -> str:
    """
    Clean and normalize presentation text
    
    Steps:
    1. Lowercase
    2. Remove boilerplate phrases
    3. Remove extra whitespace
    4. Remove special characters (keep basic punctuation)
    """
    
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove boilerplate patterns
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def calculate_readability(text: str) -> Dict[str, float]:
    """
    Calculate multiple readability metrics
    
    Returns:
    - flesch_reading_ease: 0-100 (higher = easier)
        90-100: Very Easy (5th grade)
        60-70: Standard (8th-9th grade)
        0-30: Very Difficult (college graduate)
    - flesch_kincaid_grade: Grade level (e.g., 12.0 = 12th grade)
    - gunning_fog: Years of education needed
    """
    
    if not text or len(text.split()) < 3:
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0,
            'gunning_fog': 0.0
        }
    
    try:
        flesch_ease = textstat.flesch_reading_ease(text)
        flesch_grade = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)
        
        # Normalize scores (prevent negative values)
        flesch_ease = max(0.0, min(100.0, flesch_ease))
        flesch_grade = max(0.0, flesch_grade)
        gunning_fog = max(0.0, gunning_fog)
        
        return {
            'flesch_reading_ease': round(flesch_ease, 2),
            'flesch_kincaid_grade': round(flesch_grade, 2),
            'gunning_fog': round(gunning_fog, 2)
        }
    
    except Exception as e:
        logger.warning(f"Readability calculation failed: {str(e)}")
        return {
            'flesch_reading_ease': 50.0,  # Default to "Standard"
            'flesch_kincaid_grade': 8.0,
            'gunning_fog': 8.0
        }


def calculate_lexical_richness(text: str) -> float:
    """
    Calculate Type-Token Ratio (TTR)
    
    TTR = (unique words / total words)
    
    Higher ratio = more diverse vocabulary
    - 0.0-0.3: Low diversity
    - 0.3-0.5: Moderate diversity
    - 0.5+: High diversity
    """
    
    if not text:
        return 0.0
    
    # Tokenize (split into words)
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    if len(words) < 3:
        return 0.0
    
    # Calculate TTR
    unique_words = len(set(words))
    total_words = len(words)
    
    ttr = unique_words / total_words
    
    return round(ttr, 3)


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    Extract key terms using YAKE algorithm
    
    YAKE = Yet Another Keyword Extractor
    Unsupervised, domain-independent keyword extraction
    """
    
    if not text or len(text.split()) < 5:
        return []
    
    try:
        # Initialize YAKE
        kw_extractor = yake.KeywordExtractor(
            lan="en",                    # Language
            n=2,                         # Max n-gram size (1-2 words)
            dedupLim=0.7,               # Deduplication threshold
            top=max_keywords,           # Number of keywords
            features=None
        )
        
        # Extract keywords
        keywords = kw_extractor.extract_keywords(text)
        
        # Return just the keyword strings (not scores)
        return [kw[0] for kw in keywords]
    
    except Exception as e:
        logger.warning(f"Keyword extraction failed: {str(e)}")
        return []


def count_filler_words(text: str) -> int:
    """
    Count filler words and phrases
    
    Filler words reduce presentation quality
    """
    
    if not text:
        return 0
    
    text_lower = text.lower()
    
    count = 0
    for filler in FILLER_WORDS:
        # Count occurrences
        count += text_lower.count(filler)
    
    return count


def get_readability_interpretation(score: float) -> str:
    """Get human-readable interpretation of Flesch Reading Ease score"""
    
    if score >= 90:
        return "Very Easy (5th grade)"
    elif score >= 80:
        return "Easy (6th grade)"
    elif score >= 70:
        return "Fairly Easy (7th grade)"
    elif score >= 60:
        return "Standard (8-9th grade)"
    elif score >= 50:
        return "Fairly Difficult (10-12th grade)"
    elif score >= 30:
        return "Difficult (College)"
    else:
        return "Very Difficult (College graduate)"


def get_richness_interpretation(score: float) -> str:
    """Get human-readable interpretation of lexical richness"""
    
    if score >= 0.6:
        return "Excellent vocabulary diversity"
    elif score >= 0.5:
        return "High vocabulary diversity"
    elif score >= 0.4:
        return "Good vocabulary diversity"
    elif score >= 0.3:
        return "Moderate vocabulary diversity"
    else:
        return "Low vocabulary diversity"
