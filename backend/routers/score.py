"""
📊 Scoring Router
Get scoring details and breakdowns
"""

from fastapi import APIRouter, HTTPException
from utils.file_handler import load_analysis_result
from core.logging import logger

router = APIRouter()


@router.get("/score/{analysis_id}")
async def get_scores(analysis_id: str):
    """
    Get comprehensive scoring breakdown
    
    Returns:
        - overall_score: Final 0-100 score
        - component_scores: Breakdown by category
        - per_slide_scores: Individual slide scores
        - bloom_distribution: Knowledge depth analysis
    """
    
    try:
        result = load_analysis_result(analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found. Please analyze first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load scores: {str(e)}")
    
    return {
        "analysis_id": analysis_id,
        "overall_score": result.overall_score.final_score,
        "letter_grade": result.overall_score.letter_grade,
        "component_scores": {
            "clarity": result.overall_score.avg_clarity,
            "structure": result.overall_score.avg_structure,
            "depth": result.overall_score.avg_depth,
            "design": result.overall_score.avg_design,
            "readability": result.overall_score.avg_readability,
            "coherence": result.overall_score.avg_coherence,
            "blooms": result.overall_score.avg_blooms_score
        },
        "penalties": {
            "redundancy": result.overall_score.total_redundancy_penalty
        },
        "bloom_distribution": result.overall_score.bloom_distribution,
        "per_slide_scores": [
            {
                "slide_number": score.slide_number,
                "final_score": score.final_score,
                "clarity": score.clarity_score,
                "structure": score.structure_score,
                "depth": score.depth_score,
                "design": score.design_score,
                "ai_feedback": score.ai_feedback
            }
            for score in result.scores
        ],
        "strengths": result.strengths.items,
        "weaknesses": result.weaknesses.items,
        "missing_sections": result.missing_sections.sections
    }


@router.get("/score/{analysis_id}/slide/{slide_number}")
async def get_slide_score(analysis_id: str, slide_number: int):
    """Get detailed score for specific slide"""
    
    try:
        result = load_analysis_result(analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Find slide
    slide = next((s for s in result.slides if s.slide_number == slide_number), None)
    features = next((f for f in result.features if f.slide_number == slide_number), None)
    score = next((s for s in result.scores if s.slide_number == slide_number), None)
    
    if not slide or not features or not score:
        raise HTTPException(status_code=404, detail=f"Slide {slide_number} not found")
    
    return {
        "slide_number": slide_number,
        "content": {
            "title": slide.title,
            "body_text": slide.body_text,
            "word_count": slide.total_words,
            "image_count": slide.image_count
        },
        "features": {
            "section": slide.section,
            "bloom_level": features.bloom_level,
            "readability": features.readability_score,
            "coherence": features.coherence_score,
            "is_redundant": features.is_redundant,
            "layout_quality": features.layout_quality_score
        },
        "scores": {
            "final_score": score.final_score,
            "clarity": score.clarity_score,
            "structure": score.structure_score,
            "depth": score.depth_score,
            "design": score.design_score,
            "readability": score.readability_score,
            "coherence": score.coherence_score
        },
        "ai_feedback": score.ai_feedback
    }


@router.get("/score/{analysis_id}/summary")
async def get_score_summary(analysis_id: str):
    """Get concise score summary"""
    
    try:
        result = load_analysis_result(analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return {
        "analysis_id": analysis_id,
        "overall_score": result.overall_score.final_score,
        "letter_grade": result.overall_score.letter_grade,
        "total_slides": result.metadata.total_slides,
        "strengths_count": len(result.strengths.items),
        "weaknesses_count": len(result.weaknesses.items),
        "top_strength": result.strengths.items[0] if result.strengths.items else None,
        "top_weakness": result.weaknesses.items[0] if result.weaknesses.items else None,
        "bloom_level_mode": max(result.overall_score.bloom_distribution, 
                                key=result.overall_score.bloom_distribution.get) 
                           if result.overall_score.bloom_distribution else "unknown"
    }
