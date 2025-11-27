"""
� Milestone 4: Section Classification Service
Classifies slides into presentation sections using hybrid approach:
- Rule-based keyword matching
- BERT semantic embeddings 
- Cosine similarity scoring
"""

from typing import List, Dict
from models.slide import Slide
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from core.logging import logger

# Section templates with enhanced keywords, phrases, and semantic descriptions
SECTION_TEMPLATES = {
    "introduction": {
        "keywords": ["introduction", "overview", "agenda", "outline", "welcome", "objectives", "goals", "about", "today", "we will", "purpose"],
        "phrases": ["welcome to", "today we", "this presentation", "our agenda", "key objectives", "what we'll cover"],
        "title_patterns": ["introduction", "overview", "agenda", "outline"],
        "description": "Introduction and opening slide presenting the topic, agenda, objectives or overview of the presentation",
        "weight_multiplier": 1.2  # Boost for common first slide
    },
    "background": {
        "keywords": ["background", "context", "history", "motivation", "problem", "challenge", "why", "issue", "current state", "situation"],
        "phrases": ["the problem", "background information", "current situation", "why this matters", "the challenge", "problem statement"],
        "title_patterns": ["background", "problem", "challenge", "motivation", "context"],
        "description": "Background and context information explaining the situation, history, problem statement, or motivation for the topic",
        "weight_multiplier": 1.1
    },
    "methodology": {
        "keywords": ["methodology", "methods", "approach", "how", "process", "steps", "procedure", "implementation", "solution", "design", "architecture"],
        "phrases": ["our approach", "the solution", "how we", "our method", "step by step", "we developed", "implementation details"],
        "title_patterns": ["methodology", "approach", "solution", "method", "how we", "implementation"],
        "description": "Methodology and approach explaining the solution, methods, process, technical approach or implementation details used",
        "weight_multiplier": 1.15
    },
    "results": {
        "keywords": ["results", "findings", "data", "analysis", "performance", "outcomes", "metrics", "achieved", "statistics", "numbers", "success"],
        "phrases": ["the results", "our findings", "we achieved", "performance metrics", "experimental results", "key findings", "data shows"],
        "title_patterns": ["results", "findings", "outcomes", "performance", "data", "experimental"],
        "description": "Results and findings presenting quantitative data, performance metrics, experimental outcomes, or analytical findings",
        "weight_multiplier": 1.15
    },
    "discussion": {
        "keywords": ["discussion", "implications", "insights", "interpretation", "significance", "meaning", "analysis", "what this means"],
        "phrases": ["key insights", "this means", "the implications", "what we learned", "significance of", "interpretation"],
        "title_patterns": ["discussion", "insights", "analysis", "implications"],
        "description": "Discussion and analysis interpreting results, deriving insights, and explaining implications or significance",
        "weight_multiplier": 1.1
    },
    "conclusion": {
        "keywords": ["conclusion", "summary", "takeaway", "recap", "closing", "in summary", "final", "to conclude", "wrapping up", "key points"],
        "phrases": ["in conclusion", "to summarize", "key takeaways", "in summary", "final thoughts", "wrapping up", "to recap"],
        "title_patterns": ["conclusion", "summary", "takeaways", "recap", "closing", "final"],
        "description": "Conclusion and summary slides wrapping up key points, main takeaways, or final thoughts",
        "weight_multiplier": 1.2  # Boost for common last slide
    },
    "references": {
        "keywords": ["references", "bibliography", "citations", "sources", "further reading", "works cited", "bibliography"],
        "phrases": ["further reading", "works cited", "reference list"],
        "title_patterns": ["references", "bibliography", "citations", "sources"],
        "description": "References slide listing sources, citations, bibliography, or further reading materials",
        "weight_multiplier": 1.0
    },
    "questions": {
        "keywords": ["questions", "q&a", "thank you", "contact", "discussion", "reach us", "email", "thanks"],
        "phrases": ["thank you", "any questions", "questions?", "contact us", "reach us", "get in touch", "happy to answer"],
        "title_patterns": ["questions", "q&a", "thank you", "thanks", "contact"],
        "description": "Questions and closing slide inviting audience interaction, showing contact information, or thanking the audience",
        "weight_multiplier": 1.2  # Boost for common last slide
    }
}

# Global model instance (loaded lazily)
_model = None


def get_model():
    """Lazy load the sentence transformer model"""
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers model...")
        try:
            # Use a lightweight model optimized for semantic similarity
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return _model


def keyword_score(text: str, keywords: List[str]) -> float:
    """
    Calculate keyword matching score with fuzzy matching
    
    Args:
        text: Text to analyze
        keywords: List of keywords to match
        
    Returns:
        Score between 0 and 1
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    matches = sum(1 for keyword in keywords if keyword in text_lower)
    
    # Normalize by keyword count, but cap at 1.0
    score = matches / max(len(keywords) * 0.3, 1)  # Only need 30% of keywords for full score
    return min(score, 1.0)


def phrase_score(text: str, phrases: List[str]) -> float:
    """
    Calculate phrase matching score for multi-word patterns
    
    Args:
        text: Text to analyze
        phrases: List of phrases to match
        
    Returns:
        Score between 0 and 1
    """
    if not text or not phrases:
        return 0.0
    
    text_lower = text.lower()
    matches = sum(1 for phrase in phrases if phrase in text_lower)
    
    # Phrases are more specific, so give higher weight
    score = matches / max(len(phrases) * 0.25, 1)  # Only need 25% of phrases
    return min(score, 1.0)


def title_score(title: str, patterns: List[str]) -> float:
    """
    Calculate title pattern matching score
    
    Args:
        title: Slide title
        patterns: List of title patterns to match
        
    Returns:
        Score between 0 and 1 (higher weight for title matches)
    """
    if not title or not patterns:
        return 0.0
    
    title_lower = title.lower()
    
    # Exact match in title is very strong signal
    for pattern in patterns:
        if pattern in title_lower:
            return 1.0
    
    return 0.0


def semantic_score(text: str, description: str, model: SentenceTransformer) -> float:
    """
    Calculate semantic similarity using BERT embeddings
    
    Args:
        text: Text to analyze
        description: Section description template
        model: Sentence transformer model
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    if not text or not description:
        return 0.0
    
    try:
        # Generate embeddings
        text_embedding = model.encode([text])
        template_embedding = model.encode([description])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(text_embedding, template_embedding)[0][0]
        
        # Normalize to 0-1 range (cosine similarity is -1 to 1)
        return float((similarity + 1) / 2)
    
    except Exception as e:
        logger.warning(f"Semantic scoring failed: {e}")
        return 0.0


def classify_slide(slide: Slide, model: SentenceTransformer) -> Dict[str, float]:
    """
    Classify a single slide into sections using enhanced multi-signal approach
    
    Args:
        slide: Slide to classify
        model: Sentence transformer model
        
    Returns:
        Dictionary of section scores
    """
    # Combine title and body for classification
    title = slide.title or ""
    body = slide.body_text or ""
    full_text = f"{title} {body}".strip()
    
    if not full_text:
        logger.debug(f"Slide {slide.slide_number} has no text, skipping classification")
        return {}
    
    scores = {}
    
    for section, template in SECTION_TEMPLATES.items():
        # Multi-signal scoring approach
        semantic = semantic_score(full_text, template["description"], model)
        keyword = keyword_score(full_text, template["keywords"])
        phrase = phrase_score(full_text, template.get("phrases", []))
        title_match = title_score(title, template.get("title_patterns", []))
        
        # Weighted combination with title getting highest priority
        # Title: 30%, Semantic: 40%, Keywords: 20%, Phrases: 10%
        base_score = (0.30 * title_match) + (0.40 * semantic) + (0.20 * keyword) + (0.10 * phrase)
        
        # Apply weight multiplier for certain sections
        weight = template.get("weight_multiplier", 1.0)
        final_score = base_score * weight
        
        scores[section] = min(final_score, 1.0)  # Cap at 1.0
        
        logger.debug(
            f"Slide {slide.slide_number} - {section}: "
            f"title={title_match:.3f}, semantic={semantic:.3f}, "
            f"keyword={keyword:.3f}, phrase={phrase:.3f}, "
            f"final={scores[section]:.3f}"
        )
    
    return scores


def classify_slides(slides: List[Slide]) -> List[Slide]:
    """
    Classify all slides in a presentation
    
    Args:
        slides: List of slides to classify
        
    Returns:
        List of slides with section information added
    """
    logger.info(f"Starting section classification for {len(slides)} slides")
    
    try:
        # Load model
        model = get_model()
        
        for slide in slides:
            # Get scores for all sections
            scores = classify_slide(slide, model)
            
            if scores:
                # Assign the highest-scoring section
                best_section = max(scores.items(), key=lambda x: x[1])
                slide.section = best_section[0]
                slide.section_confidence = best_section[1]
                
                logger.info(f"Slide {slide.slide_number}: classified as '{slide.section}' (confidence: {slide.section_confidence:.2f})")
            else:
                slide.section = "unknown"
                slide.section_confidence = 0.0
                logger.warning(f"Slide {slide.number}: could not classify (no text)")
        
        # Post-processing: Use position heuristics
        _apply_position_heuristics(slides)
        
        logger.info("Section classification complete")
        return slides
    
    except Exception as e:
        logger.error(f"Section classification failed: {e}")
        # Return slides unchanged on error
        return slides


def _apply_position_heuristics(slides: List[Slide]):
    """
    Apply enhanced position-based heuristics to improve classification
    
    - First slide is very likely introduction
    - Last slide is very likely conclusion/questions
    - Results typically appear in second half
    - Methodology appears before results
    """
    if not slides:
        return
    
    total_slides = len(slides)
    
    # First slide heuristic - very strong signal
    if slides[0].section != "introduction" or slides[0].section_confidence < 0.65:
        first_text = f"{slides[0].title or ''} {slides[0].body_text or ''}".lower()
        
        # Strong introduction signals
        intro_signals = ["welcome", "introduction", "agenda", "overview", "outline", "objectives", "today"]
        has_intro_signal = any(kw in first_text for kw in intro_signals)
        
        # Most first slides are introductions, boost confidence
        if has_intro_signal or slides[0].section_confidence < 0.5:
            logger.info(f"Slide 1: Boosting 'introduction' classification (position heuristic)")
            slides[0].section = "introduction"
            slides[0].section_confidence = 0.85 if has_intro_signal else 0.70
    
    # Last slide heuristic - very strong signal
    if total_slides > 1:
        last_slide = slides[-1]
        last_text = f"{last_slide.title or ''} {last_slide.body_text or ''}".lower()
        
        # Check for questions/thank you slide
        question_signals = ["thank you", "thanks", "questions", "q&a", "contact", "reach us"]
        conclusion_signals = ["conclusion", "summary", "takeaway", "recap", "in summary", "final"]
        
        has_question_signal = any(kw in last_text for kw in question_signals)
        has_conclusion_signal = any(kw in last_text for kw in conclusion_signals)
        
        if has_question_signal and (last_slide.section != "questions" or last_slide.section_confidence < 0.65):
            logger.info(f"Slide {total_slides}: Boosting 'questions' classification (position heuristic)")
            last_slide.section = "questions"
            last_slide.section_confidence = 0.85
        elif has_conclusion_signal and (last_slide.section != "conclusion" or last_slide.section_confidence < 0.65):
            logger.info(f"Slide {total_slides}: Boosting 'conclusion' classification (position heuristic)")
            last_slide.section = "conclusion"
            last_slide.section_confidence = 0.85
        elif last_slide.section_confidence < 0.5:
            # Default: last slides are usually questions or conclusion
            if "?" in last_text or len(last_text) < 100:
                last_slide.section = "questions"
                last_slide.section_confidence = 0.70
            else:
                last_slide.section = "conclusion"
                last_slide.section_confidence = 0.70
    
    # Second slide often provides background/problem
    if total_slides > 2 and slides[1].section_confidence < 0.55:
        second_text = f"{slides[1].title or ''} {slides[1].body_text or ''}".lower()
        background_signals = ["problem", "challenge", "background", "context", "motivation", "why"]
        
        if any(kw in second_text for kw in background_signals):
            logger.info(f"Slide 2: Boosting 'background' classification (position heuristic)")
            if slides[1].section == "background":
                slides[1].section_confidence = max(slides[1].section_confidence, 0.70)
    
    # Confidence boost for all low-confidence classifications
    for slide in slides:
        if slide.section_confidence < 0.50:
            # Boost confidence slightly to account for natural language variation
            slide.section_confidence = min(slide.section_confidence * 1.25, 0.65)
            logger.debug(f"Slide {slide.slide_number}: Boosted low confidence to {slide.section_confidence:.2f}")
