"""
🎯 Milestone 5: Feature Extraction Service
Extracts advanced features from presentations:
- Semantic density and coherence (SBERT)
- Content redundancy detection
- Layout quality metrics
- Bloom's Taxonomy cognitive levels
"""

from typing import List, Dict, Tuple
from models.slide import Slide
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from core.logging import logger

# Bloom's Taxonomy verb patterns (cognitive levels from lower to higher)
# EXPANDED for universal detection across all presentation types
BLOOMS_TAXONOMY = {
    "remember": {
        "level": 1,
        "verbs": ["define", "list", "name", "identify", "recall", "recognize", "state", "describe", "label", "match", "select", "know", "memorize", "repeat", "record", "locate", "find"],
        "description": "Recall facts and basic concepts"
    },
    "understand": {
        "level": 2,
        "verbs": ["explain", "describe", "discuss", "summarize", "interpret", "classify", "compare", "contrast", "demonstrate", "illustrate", "clarify", "paraphrase", "represent", "translate", "express", "indicate", "locate", "recognize", "report", "review", "tell"],
        "description": "Explain ideas or concepts"
    },
    "apply": {
        "level": 3,
        "verbs": ["apply", "execute", "implement", "use", "solve", "demonstrate", "calculate", "complete", "show", "modify", "operate", "practice", "predict", "prepare", "produce", "relate", "schedule", "sketch", "utilize", "employ", "perform", "monitor", "deploy"],
        "description": "Use information in new situations"
    },
    "analyze": {
        "level": 4,
        "verbs": ["analyze", "differentiate", "distinguish", "examine", "experiment", "question", "test", "investigate", "compare", "contrast", "diagnose", "inspect", "survey", "detect", "discover", "identify", "categorize", "measure", "assess", "evaluate", "optimize"],
        "description": "Draw connections among ideas"
    },
    "evaluate": {
        "level": 5,
        "verbs": ["evaluate", "judge", "assess", "critique", "justify", "argue", "defend", "support", "rate", "recommend", "conclude", "measure", "predict", "prioritize", "prove", "validate", "verify", "determine", "grade", "rank"],
        "description": "Justify a decision or course of action"
    },
    "create": {
        "level": 6,
        "verbs": ["create", "design", "develop", "construct", "produce", "invent", "devise", "formulate", "propose", "plan", "build", "compose", "generate", "integrate", "modify", "organize", "originate", "assemble", "innovate", "establish", "improve"],
        "description": "Produce new or original work"
    }
}

# Global model instance (shared with classification service)
_model = None


def get_model():
    """Lazy load the sentence transformer model"""
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers model for feature extraction...")
        try:
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return _model


def calculate_semantic_density(slides: List[Slide]) -> Dict[str, float]:
    """
    Calculate semantic density - how much meaning is packed into the content
    Higher density = more concepts per slide
    
    Args:
        slides: List of slides to analyze
        
    Returns:
        Dictionary with overall and per-slide semantic density scores
    """
    logger.info("Calculating semantic density...")
    
    try:
        model = get_model()
        
        # Extract all sentences from slides
        all_sentences = []
        slide_sentences = []
        
        for slide in slides:
            text = f"{slide.title or ''} {slide.body_text or ''}".strip()
            
            # Split into sentences (simple approach)
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
            
            if sentences:
                slide_sentences.append(sentences)
                all_sentences.extend(sentences)
            else:
                slide_sentences.append([])
        
        if len(all_sentences) < 2:
            logger.warning("Not enough sentences for semantic density analysis")
            return {"overall_density": 0.0, "per_slide": [0.0] * len(slides)}
        
        # Generate embeddings for all sentences
        embeddings = model.encode(all_sentences)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Semantic density = 1 - average similarity (more diverse = higher density)
        # Exclude diagonal (self-similarity = 1)
        mask = np.ones_like(similarities) - np.eye(len(similarities))
        avg_similarity = (similarities * mask).sum() / mask.sum()
        overall_density = 1 - avg_similarity
        
        # Per-slide density
        per_slide_density = []
        sentence_idx = 0
        
        for sentences in slide_sentences:
            if len(sentences) < 2:
                per_slide_density.append(0.0)
                sentence_idx += len(sentences)
                continue
            
            # Get embeddings for this slide's sentences
            slide_embeddings = embeddings[sentence_idx:sentence_idx + len(sentences)]
            slide_sim = cosine_similarity(slide_embeddings)
            
            # Calculate density for this slide
            slide_mask = np.ones_like(slide_sim) - np.eye(len(slide_sim))
            slide_avg_sim = (slide_sim * slide_mask).sum() / slide_mask.sum() if slide_mask.sum() > 0 else 0
            per_slide_density.append(1 - slide_avg_sim)
            
            sentence_idx += len(sentences)
        
        logger.info(f"Semantic density: overall={overall_density:.3f}, avg_per_slide={np.mean(per_slide_density):.3f}")
        
        return {
            "overall_density": float(overall_density),
            "per_slide": per_slide_density
        }
    
    except Exception as e:
        logger.error(f"Semantic density calculation failed: {e}")
        return {"overall_density": 0.0, "per_slide": [0.0] * len(slides)}


def detect_redundancy(slides: List[Slide], threshold: float = 0.85) -> Dict:
    """
    Detect redundant content across slides using semantic similarity
    
    Args:
        slides: List of slides to analyze
        threshold: Similarity threshold for redundancy (default 0.85)
        
    Returns:
        Dictionary with redundancy information
    """
    logger.info(f"Detecting content redundancy (threshold={threshold})...")
    
    try:
        model = get_model()
        
        # Get full text for each slide
        slide_texts = []
        for slide in slides:
            text = f"{slide.title or ''} {slide.body_text or ''}".strip()
            slide_texts.append(text if text else "empty")
        
        if len(slide_texts) < 2:
            return {"redundant_pairs": [], "redundancy_score": 0.0}
        
        # Generate embeddings
        embeddings = model.encode(slide_texts)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Find redundant pairs
        redundant_pairs = []
        for i in range(len(slides)):
            for j in range(i + 1, len(slides)):
                sim = similarities[i][j]
                if sim >= threshold:
                    redundant_pairs.append({
                        "slide_1": i + 1,
                        "slide_2": j + 1,
                        "similarity": float(sim),
                        "title_1": slides[i].title or "Untitled",
                        "title_2": slides[j].title or "Untitled"
                    })
        
        # Overall redundancy score (average of high similarities)
        high_sims = [similarities[i][j] for i in range(len(slides)) 
                     for j in range(i + 1, len(slides)) if similarities[i][j] >= threshold]
        redundancy_score = float(np.mean(high_sims)) if high_sims else 0.0
        
        logger.info(f"Found {len(redundant_pairs)} redundant pairs, score={redundancy_score:.3f}")
        
        return {
            "redundant_pairs": redundant_pairs,
            "redundancy_score": redundancy_score,
            "total_comparisons": len(slides) * (len(slides) - 1) // 2
        }
    
    except Exception as e:
        logger.error(f"Redundancy detection failed: {e}")
        return {"redundant_pairs": [], "redundancy_score": 0.0}


def analyze_layout_quality(slides: List[Slide]) -> Dict:
    """
    Analyze layout quality metrics
    
    Args:
        slides: List of slides to analyze
        
    Returns:
        Dictionary with layout quality metrics
    """
    logger.info("Analyzing layout quality...")
    
    try:
        metrics = {
            "text_to_visual_ratio": [],
            "whitespace_balance": [],
            "text_density": [],
            "image_count": []
        }
        
        for slide in slides:
            # Text to visual ratio
            text_elements = len(slide.text_boxes)
            visual_elements = len(slide.images)
            total = text_elements + visual_elements
            
            text_ratio = text_elements / total if total > 0 else 1.0
            metrics["text_to_visual_ratio"].append(text_ratio)
            
            # Whitespace balance (inverse of layout density)
            whitespace = 1.0 - slide.layout_density
            metrics["whitespace_balance"].append(whitespace)
            
            # Text density (words per text box)
            text_density = slide.total_words / max(text_elements, 1)
            metrics["text_density"].append(text_density)
            
            # Image count
            metrics["image_count"].append(visual_elements)
        
        # Calculate averages and quality scores
        avg_text_ratio = np.mean(metrics["text_to_visual_ratio"])
        avg_whitespace = np.mean(metrics["whitespace_balance"])
        avg_text_density = np.mean(metrics["text_density"])
        avg_images = np.mean(metrics["image_count"])
        
        # Quality score (0-100)
        # Ideal: 60-80% text, 40-60% whitespace, 10-30 words per box, 1-3 images
        text_score = 100 * (1 - abs(0.7 - avg_text_ratio) / 0.7)
        whitespace_score = 100 * (1 - abs(0.5 - avg_whitespace) / 0.5)
        density_score = 100 * (1 - min(abs(20 - avg_text_density) / 20, 1.0))
        image_score = 100 * (1 - min(abs(2 - avg_images) / 2, 1.0))
        
        overall_quality = (text_score + whitespace_score + density_score + image_score) / 4
        
        logger.info(f"Layout quality score: {overall_quality:.1f}/100")
        
        return {
            "overall_quality": float(overall_quality),
            "avg_text_to_visual_ratio": float(avg_text_ratio),
            "avg_whitespace_balance": float(avg_whitespace),
            "avg_text_density": float(avg_text_density),
            "avg_images_per_slide": float(avg_images),
            "per_slide_metrics": metrics
        }
    
    except Exception as e:
        logger.error(f"Layout analysis failed: {e}")
        return {"overall_quality": 0.0}


def classify_blooms_taxonomy(slides: List[Slide]) -> Dict:
    """
    Classify slides by Bloom's Taxonomy cognitive levels using keyword matching
    
    NOTE: OpenAI integration disabled to avoid rate limit errors.
    Uses keyword-based classification instead (fast, reliable, FREE).
    
    Args:
        slides: List of slides to analyze
        
    Returns:
        Dictionary with Bloom's taxonomy classification
    """
    logger.info("Classifying cognitive levels (Bloom's Taxonomy) using keywords...")
    
    # Always use keyword-based fallback (no OpenAI to avoid quota errors)
    return _fallback_blooms_taxonomy(slides)


def _classify_slide_keywords(slide: Slide) -> Dict:
    """Fallback keyword-based classification for a single slide"""
    text = f"{slide.title or ''} {slide.body_text or ''}".lower()
    
    verb_matches = {}
    for level, data in BLOOMS_TAXONOMY.items():
        matches = sum(1 for verb in data["verbs"] if verb in text)
        if matches > 0:
            verb_matches[level] = matches
    
    if verb_matches:
        weighted_scores = {
            level: matches * BLOOMS_TAXONOMY[level]["level"]
            for level, matches in verb_matches.items()
        }
        best_level = max(weighted_scores.items(), key=lambda x: x[1])[0]
    else:
        best_level = "understand"
    
    return {
        "slide_number": slide.slide_number,
        "level": best_level,
        "level_number": BLOOMS_TAXONOMY[best_level]["level"],
        "description": BLOOMS_TAXONOMY[best_level]["description"],
        "confidence": 0.5
    }


def _fallback_blooms_taxonomy(slides: List[Slide]) -> Dict:
    """Fallback keyword-based Bloom's taxonomy classification"""
    logger.info("Using keyword-based Bloom's taxonomy (fallback mode)")
    
    slide_levels = []
    level_distribution = {level: 0 for level in BLOOMS_TAXONOMY.keys()}
    
    for slide in slides:
        level_info = _classify_slide_keywords(slide)
        slide_levels.append(level_info)
        level_distribution[level_info["level"]] += 1
    
    avg_level = np.mean([s["level_number"] for s in slide_levels])
    
    level_numbers = [s["level_number"] for s in slide_levels]
    progression_score = 0.0
    if len(level_numbers) > 1:
        increases = sum(1 for i in range(len(level_numbers) - 1) 
                      if level_numbers[i + 1] >= level_numbers[i])
        progression_score = increases / (len(level_numbers) - 1)
    
    return {
        "per_slide": slide_levels,
        "distribution": level_distribution,
        "average_level": float(avg_level),
        "average_confidence": 0.5,
        "progression_score": float(progression_score),
        "highest_level": max(s["level_number"] for s in slide_levels),
        "lowest_level": min(s["level_number"] for s in slide_levels)
    }


def extract_features(slides: List[Slide]) -> Dict:
    """
    Extract all advanced features from slides
    Main entry point for Milestone 5
    
    Args:
        slides: List of slides to analyze
        
    Returns:
        Dictionary with all extracted features
    """
    logger.info(f"Starting feature extraction for {len(slides)} slides")
    
    try:
        features = {
            "semantic_density": calculate_semantic_density(slides),
            "redundancy": detect_redundancy(slides),
            "layout_quality": analyze_layout_quality(slides),
            "blooms_taxonomy": classify_blooms_taxonomy(slides)
        }
        
        logger.info("Feature extraction complete")
        return features
    
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return {}
