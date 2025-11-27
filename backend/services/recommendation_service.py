"""
Milestone 8: Recommendations Engine
Generates actionable improvement suggestions based on analysis
"""

from typing import List, Dict, Tuple
from models.slide import Slide
from core.logging import logger


def generate_recommendations(
    slides: List[Slide],
    scoring_result: Dict,
    features: Dict = None,
    max_recommendations: int = 20
) -> Dict:
    """
    Generate prioritized recommendations for presentation improvement
    Main entry point for Milestone 8
    
    Args:
        slides: List of slides
        scoring_result: Complete scoring results from Milestone 7
        max_recommendations: Maximum number of recommendations to return
        
    Returns:
        Dictionary with prioritized recommendations
    """
    logger.info(f"Generating recommendations for {len(slides)} slides")
    
    recommendations = []
    
    # Analyze each component and generate targeted recommendations
    component_scores = scoring_result.get("component_scores", {})
    feature_analysis = scoring_result.get("feature_analysis", {})
    ai_eval = scoring_result.get("ai_evaluation", {})
    slide_scores = scoring_result.get("slide_scores", [])
    
    # 1. Readability recommendations
    if component_scores.get("readability", 100) < 65:
        recommendations.extend(_generate_readability_recommendations(slides))
    
    # 2. Structure recommendations
    if component_scores.get("structure", 100) < 75:
        recommendations.extend(_generate_structure_recommendations(slides))
    
    # 3. Clarity recommendations
    if component_scores.get("clarity", 100) < 75:
        recommendations.extend(_generate_clarity_recommendations(slides))
    
    # 4. Design/Layout recommendations
    if component_scores.get("design", 100) < 70:
        recommendations.extend(_generate_design_recommendations(slides, feature_analysis))
    
    # 5. Depth/Engagement recommendations
    if component_scores.get("depth", 100) < 70:
        recommendations.extend(_generate_depth_recommendations(slides, feature_analysis))
    
    # 6. Redundancy recommendations
    if feature_analysis.get("redundancy_score", 0) > 0.3:
        recommendations.extend(_generate_redundancy_recommendations(slides))
    
    # 7. Bloom's taxonomy recommendations
    if feature_analysis.get("avg_bloom_level", 6) < 3:
        recommendations.extend(_generate_cognitive_recommendations(slides))
    
    # 8. Per-slide recommendations (for worst performing slides)
    recommendations.extend(_generate_slide_specific_recommendations(slides, slide_scores))
    
    # 9. AI-based recommendations
    recommendations.extend(_extract_ai_recommendations(ai_eval))
    
    # Prioritize and deduplicate recommendations
    prioritized = _prioritize_recommendations(recommendations, component_scores, feature_analysis)
    
    # Limit to max_recommendations
    final_recommendations = prioritized[:max_recommendations]
    
    result = {
        "total_recommendations": len(final_recommendations),
        "recommendations": final_recommendations,
        "priority_breakdown": {
            "critical": len([r for r in final_recommendations if r["priority"] == "critical"]),
            "high": len([r for r in final_recommendations if r["priority"] == "high"]),
            "medium": len([r for r in final_recommendations if r["priority"] == "medium"]),
            "low": len([r for r in final_recommendations if r["priority"] == "low"])
        },
        "estimated_impact": _calculate_total_impact(final_recommendations)
    }
    
    logger.info(f"Generated {len(final_recommendations)} recommendations")
    return result


def _generate_readability_recommendations(slides: List[Slide]) -> List[Dict]:
    """Generate recommendations for improving readability"""
    recommendations = []
    
    # Check for overly complex slides
    complex_slides = [s for s in slides if s.total_words > 80]
    if complex_slides:
        recommendations.append({
            "category": "Readability",
            "issue": "Text-heavy slides",
            "recommendation": f"Simplify slides {', '.join([str(s.slide_number) for s in complex_slides[:3]])} - reduce word count to under 50 words per slide",
            "priority": "high",
            "impact": 15,
            "effort": "medium",
            "specific_slides": [s.slide_number for s in complex_slides[:3]]
        })
    
    # Check for slides that are too sparse
    sparse_slides = [s for s in slides if s.total_words < 10 and s.body_text]
    if sparse_slides:
        recommendations.append({
            "category": "Readability",
            "issue": "Insufficient content",
            "recommendation": f"Add more context to slides {', '.join([str(s.slide_number) for s in sparse_slides[:3]])} - expand key points",
            "priority": "medium",
            "impact": 10,
            "effort": "low",
            "specific_slides": [s.slide_number for s in sparse_slides[:3]]
        })
    
    return recommendations


def _generate_structure_recommendations(slides: List[Slide]) -> List[Dict]:
    """Generate recommendations for improving structure"""
    recommendations = []
    
    # Check for missing introduction
    has_intro = any(getattr(s, 'section', None) == "introduction" for s in slides[:2])
    if not has_intro:
        recommendations.append({
            "category": "Structure",
            "issue": "Missing introduction",
            "recommendation": "Add a clear introduction slide at the beginning to set context and objectives",
            "priority": "high",
            "impact": 20,
            "effort": "low",
            "specific_slides": [1]
        })
    
    # Check for missing conclusion
    has_conclusion = any(getattr(s, 'section', None) in ["conclusion", "questions"] for s in slides[-2:])
    if not has_conclusion:
        recommendations.append({
            "category": "Structure",
            "issue": "Missing conclusion",
            "recommendation": "Add a conclusion slide to summarize key takeaways and next steps",
            "priority": "high",
            "impact": 20,
            "effort": "low",
            "specific_slides": [len(slides)]
        })
    
    # Check for proper flow
    if len(slides) > 5:
        recommendations.append({
            "category": "Structure",
            "issue": "Section organization",
            "recommendation": "Consider adding transition slides between major sections to improve flow",
            "priority": "medium",
            "impact": 12,
            "effort": "low",
            "specific_slides": []
        })
    
    return recommendations


def _generate_clarity_recommendations(slides: List[Slide]) -> List[Dict]:
    """Generate recommendations for improving clarity"""
    recommendations = []
    
    # Check for missing titles
    slides_without_title = [s for s in slides if not s.title]
    if slides_without_title:
        recommendations.append({
            "category": "Clarity",
            "issue": "Missing slide titles",
            "recommendation": f"Add descriptive titles to slides {', '.join([str(s.slide_number) for s in slides_without_title])}",
            "priority": "high",
            "impact": 15,
            "effort": "low",
            "specific_slides": [s.slide_number for s in slides_without_title]
        })
    
    # Check for vague titles
    vague_titles = ["Slide", "Untitled", "Content", "Information"]
    slides_with_vague_titles = [s for s in slides if s.title and any(v.lower() in s.title.lower() for v in vague_titles)]
    if slides_with_vague_titles:
        recommendations.append({
            "category": "Clarity",
            "issue": "Vague slide titles",
            "recommendation": "Make titles more specific and descriptive to convey slide purpose clearly",
            "priority": "medium",
            "impact": 10,
            "effort": "low",
            "specific_slides": [s.slide_number for s in slides_with_vague_titles]
        })
    
    return recommendations


def _generate_design_recommendations(slides: List[Slide], features: Dict) -> List[Dict]:
    """Generate recommendations for improving design/layout"""
    recommendations = []
    
    # Check for lack of visuals
    text_only_slides = [s for s in slides if s.image_count == 0 and s.total_words > 20]
    if len(text_only_slides) > len(slides) * 0.6:
        recommendations.append({
            "category": "Design",
            "issue": "Insufficient visual elements",
            "recommendation": "Add charts, diagrams, or images to at least 40% of slides to improve visual engagement",
            "priority": "high",
            "impact": 18,
            "effort": "medium",
            "specific_slides": [s.slide_number for s in text_only_slides[:3]]
        })
    
    # Check layout quality
    layout_quality = features.get("layout_quality", 100)
    if layout_quality < 60:
        recommendations.append({
            "category": "Design",
            "issue": "Inconsistent layout",
            "recommendation": "Standardize slide layouts - maintain consistent margins, fonts, and element positioning",
            "priority": "medium",
            "impact": 15,
            "effort": "medium",
            "specific_slides": []
        })
    
    # Check for overcrowded slides
    crowded = [s for s in slides if s.total_words > 80 or (len(s.text_boxes) > 5 if hasattr(s, 'text_boxes') and s.text_boxes else False)]
    if crowded:
        recommendations.append({
            "category": "Design",
            "issue": "Overcrowded slides",
            "recommendation": "Split dense content across multiple slides for better readability",
            "priority": "high",
            "impact": 16,
            "effort": "medium",
            "specific_slides": [s.slide_number for s in crowded[:2]]
        })
    
    return recommendations


def _generate_depth_recommendations(slides: List[Slide], features: Dict) -> List[Dict]:
    """Generate recommendations for improving content depth"""
    recommendations = []
    
    avg_bloom = features.get("avg_bloom_level", 3)
    
    if avg_bloom < 3:
        recommendations.append({
            "category": "Depth",
            "issue": "Surface-level content",
            "recommendation": "Deepen analysis by including 'why' and 'how' explanations, not just 'what' descriptions",
            "priority": "high",
            "impact": 20,
            "effort": "high",
            "specific_slides": []
        })
    
    # Check for lack of examples
    has_examples = any("example" in s.body_text.lower() or "case" in s.body_text.lower() for s in slides if s.body_text)
    if not has_examples:
        recommendations.append({
            "category": "Depth",
            "issue": "Missing concrete examples",
            "recommendation": "Add real-world examples or case studies to illustrate key concepts",
            "priority": "medium",
            "impact": 15,
            "effort": "medium",
            "specific_slides": []
        })
    
    return recommendations


def _generate_redundancy_recommendations(slides: List[Slide]) -> List[Dict]:
    """Generate recommendations for reducing redundancy"""
    return [{
        "category": "Content",
        "issue": "Redundant content detected",
        "recommendation": "Remove or consolidate duplicate information across slides to improve conciseness",
        "priority": "medium",
        "impact": 12,
        "effort": "low",
        "specific_slides": []
    }]


def _generate_cognitive_recommendations(slides: List[Slide]) -> List[Dict]:
    """Generate recommendations for improving cognitive engagement"""
    return [{
        "category": "Engagement",
        "issue": "Low cognitive engagement",
        "recommendation": "Include activities that require analysis, evaluation, or creation - not just recall or understanding",
        "priority": "medium",
        "impact": 18,
        "effort": "high",
        "specific_slides": []
    }]


def _generate_slide_specific_recommendations(slides: List[Slide], slide_scores: List[Dict]) -> List[Dict]:
    """Generate recommendations for specific low-scoring slides"""
    recommendations = []
    
    if not slide_scores:
        return recommendations
    
    # Find worst 2 slides
    sorted_slides = sorted(slide_scores, key=lambda x: x["score"])
    worst_slides = sorted_slides[:2]
    
    for slide_score in worst_slides:
        if slide_score["score"] < 60:
            slide_num = slide_score["slide_number"]
            recommendations.append({
                "category": "Slide Quality",
                "issue": f"Slide {slide_num} scored {slide_score['score']:.0f}/100",
                "recommendation": f"Revise slide {slide_num}: improve clarity ({slide_score['clarity']:.0f}), structure ({slide_score['structure']:.0f}), and design ({slide_score['design']:.0f})",
                "priority": "high",
                "impact": 14,
                "effort": "medium",
                "specific_slides": [slide_num]
            })
    
    return recommendations


def _extract_ai_recommendations(ai_eval: Dict) -> List[Dict]:
    """Extract recommendations from AI evaluation"""
    recommendations = []
    
    ai_recommendations = ai_eval.get("recommendations", [])
    
    for i, rec in enumerate(ai_recommendations[:3]):  # Top 3 AI recommendations
        recommendations.append({
            "category": "AI Insight",
            "issue": "AI-identified improvement area",
            "recommendation": rec,
            "priority": "medium",
            "impact": 12,
            "effort": "medium",
            "specific_slides": []
        })
    
    return recommendations


def _prioritize_recommendations(
    recommendations: List[Dict],
    component_scores: Dict,
    features: Dict
) -> List[Dict]:
    """Prioritize and deduplicate recommendations"""
    
    # Remove duplicates based on recommendation text
    seen = set()
    unique_recs = []
    for rec in recommendations:
        rec_key = rec["recommendation"].lower()
        if rec_key not in seen:
            seen.add(rec_key)
            unique_recs.append(rec)
    
    # Sort by priority (critical > high > medium > low) and then by impact
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_recs = sorted(
        unique_recs,
        key=lambda x: (priority_order.get(x["priority"], 3), -x["impact"])
    )
    
    return sorted_recs


def _calculate_total_impact(recommendations: List[Dict]) -> Dict:
    """Calculate estimated total impact of implementing recommendations"""
    if not recommendations:
        return {"total_points": 0, "percentage": 0}
    
    total_impact = sum(r["impact"] for r in recommendations)
    
    return {
        "total_points": total_impact,
        "percentage": min(100, total_impact),  # Cap at 100%
        "description": f"Implementing all recommendations could improve score by up to {min(100, total_impact)} points"
    }
