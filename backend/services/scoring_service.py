"""
Milestone 7: Scoring Engine
Combines all metrics into final presentation score (0-100)
"""

from typing import List, Dict, Optional
from models.slide import Slide, SlideScore
from services.preprocessing_service import preprocess_text
from services.classification_service import classify_slide
from services.feature_extraction_service import extract_features
from services.ai_evaluation_service import evaluate_slides
from core.config import settings
from core.logging import logger
import statistics


async def calculate_slide_scores(slides: List[Slide], audience_type: str = "general") -> List[SlideScore]:
    """
    Calculate comprehensive scores for each slide
    
    Args:
        slides: List of slides to score
        audience_type: Target audience type for AI evaluation
        
    Returns:
        List of SlideScore objects with detailed scoring
    """
    logger.info(f"Calculating scores for {len(slides)} slides")
    
    # Step 1: Extract features for all slides (not async)
    features_result = extract_features(slides)
    
    # Step 2: Get AI evaluation
    ai_eval = await evaluate_slides(slides, audience_type)
    
    # Step 3: Score each slide
    slide_scores = []
    
    for i, slide in enumerate(slides):
        # Get slide-specific features
        blooms_per_slide = features_result.get("blooms_taxonomy", {}).get("per_slide", [])
        slide_features = blooms_per_slide[i] if i < len(blooms_per_slide) else None

        # Calculate redundancy involvement for this slide (how many redundant pairs include this slide)
        redundancy_info = features_result.get("redundancy", {})
        redundant_pairs = redundancy_info.get("redundant_pairs", [])
        total_pairs = redundancy_info.get("total_comparisons", 0) or (len(slides) * (len(slides) - 1) // 2)
        # Count occurrences where this slide is present in a redundant pair
        slide_idx = i + 1
        occurrences = sum(1 for p in redundant_pairs if p.get("slide_1") == slide_idx or p.get("slide_2") == slide_idx)
        # Normalize penalty: fraction of pairs this slide participates in multiplied by a max penalty (30)
        redundancy_penalty = 0.0
        if total_pairs > 0:
            redundancy_penalty = min(30.0, (occurrences / max(1, total_pairs)) * 30.0)
        
        # Calculate component scores
        clarity_score = _calculate_clarity_score(slide, slide_features)
        structure_score = _calculate_structure_score(slide, i, len(slides))
        depth_score = _calculate_depth_score(slide, slide_features)
        design_score = _calculate_design_score(slide, slide_features)
        readability_score = _calculate_readability_score(slide)
        coherence_score = _calculate_coherence_score(slide, slide_features)
        blooms_score = _calculate_blooms_score(slide_features)
        # fallback to previously computed redundancy_penalty if the helper returned something
        if not redundancy_penalty:
            redundancy_penalty = _calculate_redundancy_penalty(slide_features)
        
        # Calculate weighted final score
        final_score = (
            clarity_score * settings.WEIGHT_CLARITY +
            structure_score * settings.WEIGHT_STRUCTURE +
            depth_score * settings.WEIGHT_DEPTH +
            design_score * settings.WEIGHT_DESIGN +
            readability_score * settings.WEIGHT_READABILITY +
            coherence_score * settings.WEIGHT_COHERENCE +
            blooms_score * settings.WEIGHT_BLOOMS -
            redundancy_penalty * settings.WEIGHT_REDUNDANCY_PENALTY
        )
        
        # Clamp to 0-100
        final_score = max(0, min(100, final_score))
        
        slide_score = SlideScore(
            slide_number=slide.slide_number,
            clarity_score=clarity_score,
            structure_score=structure_score,
            depth_score=depth_score,
            design_score=design_score,
            readability_score=readability_score,
            coherence_score=coherence_score,
            blooms_score=blooms_score,
            redundancy_penalty=redundancy_penalty,
            final_score=final_score,
            ai_feedback={
                "content_quality": ai_eval.get("content_quality", {}),
                "clarity_coherence": ai_eval.get("clarity_coherence", {})
            }
        )
        
        slide_scores.append(slide_score)
        logger.debug(f"Slide {slide.slide_number}: Final score = {final_score:.1f}/100")
    
    return slide_scores


async def calculate_presentation_score(slides: List[Slide], audience_type: str = "general", milestone3_score: Optional[float] = None) -> Dict:
    """
    Calculate overall presentation score combining all metrics
    Main entry point for Milestone 7
    
    Args:
        slides: List of slides to score
        audience_type: Target audience type
        
    Returns:
        Dictionary with comprehensive scoring results
    """
    logger.info(f"Starting comprehensive scoring for {len(slides)} slides")
    
    try:
        # Get all component evaluations
        slide_scores = await calculate_slide_scores(slides, audience_type)
        features_result = extract_features(slides)
        ai_eval = await evaluate_slides(slides, audience_type)
        
        # Calculate overall metrics
        avg_slide_score = statistics.mean([s.final_score for s in slide_scores])
        
        # Component averages
        avg_clarity = statistics.mean([s.clarity_score for s in slide_scores])
        avg_structure = statistics.mean([s.structure_score for s in slide_scores])
        avg_depth = statistics.mean([s.depth_score for s in slide_scores])
        avg_design = statistics.mean([s.design_score for s in slide_scores])
        avg_readability = statistics.mean([s.readability_score for s in slide_scores])
        avg_coherence = statistics.mean([s.coherence_score for s in slide_scores])
        avg_blooms = statistics.mean([s.blooms_score for s in slide_scores])
        avg_redundancy = statistics.mean([s.redundancy_penalty for s in slide_scores])
        
        # NEW: Calculate presentation-level industry scores
        innovation_score = _calculate_innovation_score(slides, ai_eval)
        market_relevance_score = _calculate_market_relevance_score(slides, ai_eval)
        execution_feasibility_score = _calculate_execution_feasibility_score(slides, ai_eval)
        data_evidence_score = _calculate_data_evidence_score(slides, ai_eval)
        impact_scalability_score = _calculate_impact_scalability_score(slides, ai_eval)
        professional_quality_score = _calculate_professional_quality_score(slides, ai_eval)
        
        logger.info(f"Innovation: {innovation_score:.1f}, Market: {market_relevance_score:.1f}, "
                   f"Execution: {execution_feasibility_score:.1f}, Data: {data_evidence_score:.1f}, "
                   f"Impact: {impact_scalability_score:.1f}, Professional: {professional_quality_score:.1f}")
        
        # Calculate weighted presentation score (normalize weights)
        # INDUSTRY-LEVEL GRADING: Content Quality (55%) + Industry Metrics (40%) + Presentation (5%)
        weighted_sum = (
            # Content Quality (55% total)
            avg_depth * settings.WEIGHT_DEPTH +                    # 20%
            avg_coherence * settings.WEIGHT_COHERENCE +            # 12%
            avg_blooms * settings.WEIGHT_BLOOMS +                  # 8%
            innovation_score * settings.WEIGHT_INNOVATION +        # 15%
            
            # Industry Metrics (40% total)
            market_relevance_score * settings.WEIGHT_MARKET_RELEVANCE +         # 12%
            execution_feasibility_score * settings.WEIGHT_EXECUTION_FEASIBILITY + # 10%
            data_evidence_score * settings.WEIGHT_DATA_EVIDENCE +               # 8%
            impact_scalability_score * settings.WEIGHT_IMPACT_SCALABILITY +     # 6%
            professional_quality_score * settings.WEIGHT_PROFESSIONAL_QUALITY +  # 4%
            
            # Presentation Quality (5% total)
            avg_clarity * settings.WEIGHT_CLARITY +                # 2%
            avg_structure * settings.WEIGHT_STRUCTURE +            # 1%
            avg_design * settings.WEIGHT_DESIGN +                  # 1.5%
            avg_readability * settings.WEIGHT_READABILITY          # 0.5%
        )
        weight_total = (
            settings.WEIGHT_DEPTH + settings.WEIGHT_COHERENCE + settings.WEIGHT_BLOOMS + 
            settings.WEIGHT_INNOVATION + settings.WEIGHT_MARKET_RELEVANCE + 
            settings.WEIGHT_EXECUTION_FEASIBILITY + settings.WEIGHT_DATA_EVIDENCE + 
            settings.WEIGHT_IMPACT_SCALABILITY + settings.WEIGHT_PROFESSIONAL_QUALITY +
            settings.WEIGHT_CLARITY + settings.WEIGHT_STRUCTURE + settings.WEIGHT_DESIGN + 
            settings.WEIGHT_READABILITY
        )

        # Base presentation score normalized to 0-100
        presentation_score = weighted_sum / max(1e-6, weight_total)

        # Apply redundancy penalty (subtract percentage points)
        redundancy_penalty_points = avg_redundancy * (settings.WEIGHT_REDUNDANCY_PENALTY * 100)
        presentation_score -= redundancy_penalty_points

        # STRICT ACADEMIC PENALTIES - Multi-tier system
        hard_penalty = 0.0
        critical_failures = 0  # Count how many critical components failed
        
        # Design penalties (UI/UX is critical for presentations)
        if avg_design < settings.DESIGN_CRITICAL_THRESHOLD:
            hard_penalty += settings.DESIGN_CRITICAL_PENALTY
            critical_failures += 1
        elif avg_design < settings.DESIGN_MIN_THRESHOLD:
            hard_penalty += settings.DESIGN_PENALTY_POINTS
        
        # Readability penalties (must be readable)
        if avg_readability < settings.READABILITY_CRITICAL_THRESHOLD:
            hard_penalty += settings.READABILITY_CRITICAL_PENALTY
            critical_failures += 1
        elif avg_readability < settings.READABILITY_MIN_THRESHOLD:
            hard_penalty += settings.READABILITY_PENALTY_POINTS
        
        # Depth penalties (content quality is essential)
        if avg_depth < settings.DEPTH_CRITICAL_THRESHOLD:
            hard_penalty += settings.DEPTH_CRITICAL_PENALTY
            critical_failures += 1
        elif avg_depth < settings.DEPTH_MIN_THRESHOLD:
            hard_penalty += settings.DEPTH_PENALTY_POINTS
        
        # Structure penalties (proper organization required)
        if avg_structure < settings.STRUCTURE_MIN_THRESHOLD:
            hard_penalty += settings.STRUCTURE_PENALTY_POINTS
        
        # Visual requirements penalty (presentations need visuals!)
        total_images = sum(slide.image_count for slide in slides)
        avg_images_per_slide = total_images / len(slides) if slides else 0
        if avg_images_per_slide < settings.MIN_IMAGES_PER_SLIDE:
            hard_penalty += settings.NO_VISUAL_PENALTY

        presentation_score -= hard_penalty

        # Multi-component failure: If 2+ critical failures, apply multiplier
        if critical_failures >= 2:
            presentation_score *= settings.MULTI_FAILURE_MULTIPLIER
            logger.warning(f"Multiple critical failures ({critical_failures}) - applying {settings.MULTI_FAILURE_MULTIPLIER}x multiplier")

        # Blend with AI evaluation (minimal 5% for academic context)
        ai_score = ai_eval.get("overall_score", presentation_score)
        ai_blend = getattr(settings, 'AI_BLEND', 0.05)
        old_ppt_score = (presentation_score * (1 - ai_blend)) + (ai_score * ai_blend)
        old_ppt_score = max(0, min(100, old_ppt_score))
        
        # Combine Milestone 3 (90%) + Old PPT Scoring (10%)
        if milestone3_score is not None:
            final_score = (milestone3_score * 0.90) + (old_ppt_score * 0.10)
            logger.info(f"Combined score: Milestone 3 ({milestone3_score:.1f}) * 90% + Old PPT ({old_ppt_score:.1f}) * 10% = {final_score:.1f}")
        else:
            final_score = old_ppt_score
            logger.info(f"Using old PPT score only (Milestone 3 not available): {final_score:.1f}")
        
        final_score = max(0, min(100, final_score))
        
        # Calculate grade
        grade = _calculate_grade(final_score)
        
        # Identify strengths and weaknesses
        strengths = _identify_strengths(slide_scores, features_result, ai_eval)
        weaknesses = _identify_weaknesses(slide_scores, features_result, ai_eval)
        
        result = {
            "overall_score": final_score,
            "grade": grade,
            "slide_scores": [
                {
                    "slide_number": s.slide_number,
                    "score": s.final_score,
                    "clarity": s.clarity_score,
                    "structure": s.structure_score,
                    "depth": s.depth_score,
                    "design": s.design_score,
                    "readability": s.readability_score,
                    "coherence": s.coherence_score,
                    "blooms": s.blooms_score
                }
                for s in slide_scores
            ],
            "component_scores": {
                "clarity": avg_clarity,
                "structure": avg_structure,
                "depth": avg_depth,
                "design": avg_design,
                "readability": avg_readability,
                "coherence": avg_coherence,
                "blooms_taxonomy": avg_blooms,
                "innovation": innovation_score,
                "market_relevance": market_relevance_score,
                "execution_feasibility": execution_feasibility_score,
                "data_evidence": data_evidence_score,
                "impact_scalability": impact_scalability_score,
                "professional_quality": professional_quality_score,
                "redundancy_penalty": avg_redundancy
            },
            "feature_analysis": {
                "semantic_density": features_result.get("semantic_density", {}).get("overall_density", 0),
                "redundancy_score": features_result.get("redundancy", {}).get("redundancy_score", 0),
                "layout_quality": features_result.get("layout_quality", {}).get("overall_quality", 0),
                "avg_bloom_level": features_result.get("blooms_taxonomy", {}).get("average_level", 0)
            },
            "ai_evaluation": {
                "ai_overall_score": ai_score,
                "content_quality": ai_eval.get("content_quality", {}).get("content_quality_score", 0),
                "clarity_coherence": ai_eval.get("clarity_coherence", {}).get("clarity_score", 0),
                "audience_fit": ai_eval.get("audience_fit", {}).get("appropriateness_score", 0),
                "professional_standards": ai_eval.get("professional_standards", {}).get("standards_score", 0)
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "statistics": {
                "total_slides": len(slides),
                "avg_slide_score": avg_slide_score,
                "best_slide": max(slide_scores, key=lambda s: s.final_score).slide_number,
                "worst_slide": min(slide_scores, key=lambda s: s.final_score).slide_number,
                "score_std_dev": statistics.stdev([s.final_score for s in slide_scores]) if len(slide_scores) > 1 else 0
            }
        }
        
        logger.info(f"Scoring complete - Final: {final_score:.1f}/100 ({grade})")
        return result
        
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise


def _calculate_clarity_score(slide: Slide, features: Optional[any]) -> float:
    """
    Calculate clarity score - PROFESSIONAL PRESENTATION FOCUS
    Base score: 50 (was 70)
    Rewards clear communication for technical/professional audience
    """
    score = 50.0  # Base score for professional presentations
    
    # Title quality
    if slide.title and len(slide.title.strip()) > 5:
        score += 30  # Strong title (increased from 20)
    elif slide.title:
        score += 10  # Basic title (increased from 5)
    else:
        score -= 5  # Reduced penalty (was 8)
    
    # Word count - FLEXIBLE for technical content
    if slide.total_words > settings.CROWDED_WORD_THRESHOLD:  # > 80
        score -= 8  # Reduced penalty (was 12) - technical slides need more words
    elif slide.total_words > 60:
        score += 5  # Good for detailed content (was penalty)
    elif slide.total_words < settings.SPARSE_WORD_THRESHOLD:  # < 10
        score -= 5  # Reduced penalty (was 8)
    elif 20 <= slide.total_words <= 60:
        score += 15  # Optimal range (increased from 10)
    
    # Check for content
    if slide.body_text and len(slide.body_text.strip()) > 0:
        score += 15  # Increased from 10
    
    return max(0, min(100, score))


def _calculate_structure_score(slide: Slide, index: int, total: int) -> float:
    """
    Calculate structure score - PROFESSIONAL PRESENTATION FOCUS
    Base score: 45 (was 60)
    Rewards proper organization
    """
    score = 45.0  # Base score for professional presentations (was 60)
    
    # Section classification confidence
    if hasattr(slide, 'section_confidence') and slide.section_confidence:
        if slide.section_confidence >= 0.8:
            score += 35  # High confidence (increased from 30)
        elif slide.section_confidence >= 0.6:
            score += 25  # Medium confidence (increased from 20)
        elif slide.section_confidence >= 0.4:
            score += 15  # Some confidence (increased from 10)
    
    # Proper positioning
    if index == 0:
        if hasattr(slide, 'section') and slide.section == "introduction":
            score += 20  # Good intro (increased from 15)
    
    if index == total - 1:
        if hasattr(slide, 'section') and slide.section in ["conclusion", "questions"]:
            score += 25  # Good conclusion (increased from 20)
        else:
            score -= 5  # Reduced penalty (was 8)
    
    # Middle slides should be content (not intro/conclusion)
    if 0 < index < total - 1:
        if hasattr(slide, 'section') and slide.section in ["methodology", "results", "discussion"]:
            score += 20  # Proper content section (increased from 15)
    
    return max(0, min(100, score))


def _calculate_depth_score(slide: Slide, features: Optional[any]) -> float:
    """
    Calculate content depth score - SEMANTIC ANALYSIS IS KING
    Base score: 0 (ULTRA STRICT - prove your depth!)
    
    Evaluates:
    - Bloom's taxonomy (cognitive complexity) - MAXIMUM 60 POINTS
    - Semantic density (meaningful concepts) - MAXIMUM 30 POINTS
    - Content substance (detail level) - MAXIMUM 10 POINTS
    
    Bad presentations start at 0-30, good presentations reach 70-90!
    """
    # Make depth less brutally strict: give a reasonable base so good slides aren't scored <40 by default
    score = 30.0  # base floor for depth
    
    # 1. Bloom's taxonomy level (MAXIMUM 60 points) - PRIMARY DRIVER
    if features and isinstance(features, dict):
        level_number = features.get('level_number', 1)
        
        # Level 1-2: Basic - small uplift
        if level_number <= 2:
            score += 8
        # Level 3: Application - moderate uplift
        elif level_number == 3:
            score += 28
        # Level 4: Analysis - strong uplift
        elif level_number == 4:
            score += 48
        # Level 5-6: Evaluate/Create - max uplift
        else:
            score += 60
    
    # 2. Semantic density (MAXIMUM 30 points) - CRITICAL
    if features and isinstance(features, dict):
        semantic_density = features.get('semantic_density', 0)
        if isinstance(semantic_density, dict):
            semantic_density = semantic_density.get('overall_density', 0)
        
        # Reward semantic density more gently so useful content isn't zeroed out
        if semantic_density >= 0.75:
            score += 30
        elif semantic_density >= 0.65:
            score += 20
        elif semantic_density >= 0.55:
            score += 10
        elif semantic_density >= 0.45:
            score += 4
        else:
            score += 0
    
    # 3. Content substance (MAXIMUM 10 points) - MINOR
    if slide.total_words >= 60:
        score += 10  # Detailed
    elif slide.total_words >= 40:
        score += 6   # Good
    elif slide.total_words >= 20:
        score += 2   # Minimal
    # Below 20 words = 0 points
    
    # Ensure a reasonable floor so depth doesn't drop under ~40 unless truly empty
    final = max(40.0, min(100.0, score))
    return final


def _calculate_design_score(slide: Slide, features: Optional[any]) -> float:
    """
    Calculate design/layout score - PROFESSIONAL PRESENTATION FOCUS
    Base score: 45 (was 60)
    Visuals are important but not mandatory for every slide
    """
    score = 45.0  # Base score for professional presentations (was 60)
    
    # Images/visuals are important for engagement
    if slide.image_count == 0:
        score += 10  # Text-only OK for technical slides (was -3 penalty)
    elif slide.image_count == 1:
        score += 25  # Good - has visuals (increased from 20)
    elif slide.image_count >= 2:
        score += 30  # Excellent visual content
    
    # Text-visual balance
    if slide.total_words > 0 and slide.image_count > 0:
        # Good balance
        if 20 <= slide.total_words <= 60 and slide.image_count >= 1:
            score += 15  # Excellent balance
        else:
            score += 5   # Some balance
    elif slide.total_words > 60 and slide.image_count == 0:
        score -= 20  # Reduced penalty from 25 to 20
    
    # Check for whitespace (estimated from word count)
    if slide.total_words > 80:
        score -= 12  # Reduced penalty from 15 to 12
    
    return max(0, min(100, score))


def _calculate_readability_score(slide: Slide) -> float:
    """
    Calculate readability score - PROFESSIONAL/TECHNICAL FOCUS
    
    For college/startup/market level presentations:
    - Technical complexity is GOOD (shows depth)
    - Base score starts at 50 (not 0)
    - Graduate-level content (Flesch 0-30) is OPTIMAL for technical fields
    """
    if not slide.body_text or len(slide.body_text.strip()) < 10:
        return 50.0  # Base score for minimal content
    
    # Use preprocessing service to get Flesch readability
    result = preprocess_text(slide.body_text)
    flesch_score = result.get("readability_score", 50.0)
    grade_level = result.get("readability_grade", 12.0)
    
    # PROFESSIONAL PRESENTATION SCORING (College/Startup/Market Level):
    # Flesch 0-30 (Graduate) = EXCELLENT for technical/professional
    # Flesch 30-50 (College) = GOOD for business/academic
    # Flesch 50-70 (High School) = ACCEPTABLE for general audience
    # Flesch 70+ (Elementary) = TOO SIMPLE for professional context
    
    # Base score starts at 50 for any content but we enforce a practical floor
    base = 50
    
    # Graduate level (Flesch 0-30) - OPTIMAL for technical presentations
    if flesch_score < 30:
        # Technical/Medical/Academic - this is GOOD
        bonus = 35  # Strong bonus for depth
        return max(40.0, min(95.0, base + bonus))  # 85-95 range, floor 40
    elif 30 <= flesch_score < 50:
        bonus = 25  # Good bonus
        return max(40.0, min(85.0, base + bonus))  # 75-85 range, floor 40
    elif 50 <= flesch_score < 70:
        bonus = 15  # Moderate bonus
        return max(40.0, min(75.0, base + bonus))  # 65-75 range, floor 40
    else:
        # Elementary level (Flesch 70+) - TOO SIMPLE for professional
        bonus = 5  # Small bonus - too basic
        return max(40.0, min(65.0, base + bonus))  # 55-65 range, floor 40

    # Fallback (shouldn't be reached)
    return max(40.0, base)


def _calculate_coherence_score(slide: Slide, features: Optional[any]) -> float:
    """
    Calculate coherence score - LOGICAL DEPTH & REASONING
    Base score: 50 (professional minimum, increased from 35)
    
    Evaluates:
    - Title-content relationship
    - Logical flow and structure
    - Depth of explanation
    
    High coherence = well-organized, logically connected content
    """
    score = 50.0  # Base score increased from 35
    
    # 1. Title and content relationship (MAXIMUM 35 points)
    if slide.title and slide.body_text:
        title_len = len(slide.title.strip())
        body_len = len(slide.body_text.strip())
        
        # Strong content = excellent coherence
        if body_len > 40:
            score += 35  # Substantial content
        elif body_len > 20:
            score += 25  # Good content
        elif body_len > 10:
            score += 15  # Some content
        else:
            score += 5   # Minimal
    elif slide.title or slide.body_text:
        score += 10  # Only one present
    
    # 2. Sentence structure (MAXIMUM 15 points)
    if slide.body_text:
        has_sentences = any(char in slide.body_text for char in ['.', '!', '?'])
        
        if has_sentences:
            sentence_count = (slide.body_text.count('.') + 
                            slide.body_text.count('!') + 
                            slide.body_text.count('?'))
            
            if sentence_count >= 4:
                score += 15  # Multiple sentences (deep explanation)
            elif sentence_count >= 2:
                score += 10  # Some sentences
            else:
                score += 5   # Basic structure
    
    return max(0, min(100, score))


def _calculate_blooms_score(features: Optional[any]) -> float:
    """
    Calculate Bloom's taxonomy score - PROFESSIONAL/TECHNICAL FOCUS
    Base score: 40 (not 0)
    HIGH reward for advanced cognitive levels (4-6)
    """
    if features and isinstance(features, dict):
        level_number = features.get('level_number', 1)  # Default to lowest
        
        # PROFESSIONAL FOCUS: Levels 3-6 are EXCELLENT
        # Even levels 1-2 get base score (not penalized)
        
        # Level 1-2: Remember/Understand (basic, but OK for intro)
        if level_number <= 2:
            return 40.0  # Base score (not penalized)
        # Level 3: Apply (good for professional)
        elif level_number == 3:
            return 65.0  # Good score (increased from 50)
        # Level 4: Analyze (excellent for technical)
        elif level_number == 4:
            return 80.0  # Excellent score (new tier)
        # Level 5-6: Evaluate/Create (optimal for professional)
        else:
            return 95.0  # Maximum reward (increased from level-based)
    
    return 40.0  # Base score (was 30)


def _calculate_redundancy_penalty(features: Optional[any]) -> float:
    """Calculate redundancy penalty - presentation-level check"""
    # If caller passes a features dict containing redundancy details, use it.
    try:
        if features and isinstance(features, dict):
            # If a redundancy_score (0-1) is provided, scale to a 0-30 penalty
            red_score = features.get("redundancy_score") or features.get("redundancy", {}).get("redundancy_score")
            if red_score is not None:
                return float(red_score) * 30.0
    except Exception:
        pass
    # Default fallback
    return 0.0


def _calculate_innovation_score(slides: List[Slide], ai_eval: Dict) -> float:
    """
    Calculate innovation/startup value score - NEW METRIC
    Base score: 30 (reduced from 40 - be stricter)
    
    Evaluates:
    - Problem-solving language
    - Technical/solution keywords
    - Research/analysis depth
    - Real-world applicability
    
    Separates REAL projects (SIH, startups, research) from basic presentations!
    """
    score = 30.0  # Base reduced from 40
    
    # Get all text from slides
    all_text = " ".join([s.body_text.lower() if s.body_text else "" for s in slides])
    all_titles = " ".join([s.title.lower() if s.title else "" for s in slides])
    combined_text = all_text + " " + all_titles
    
    # 1. Problem-solving indicators (MAXIMUM 25 points)
    problem_keywords = [
        'problem', 'challenge', 'issue', 'gap', 'limitation',
        'solution', 'solve', 'address', 'resolve', 'tackle',
        'improve', 'enhance', 'optimize', 'innovate'
    ]
    problem_count = sum(1 for keyword in problem_keywords if keyword in combined_text)
    
    if problem_count >= 8:
        score += 25  # Strong problem-solving focus
    elif problem_count >= 5:
        score += 18  # Good focus
    elif problem_count >= 3:
        score += 10  # Some focus
    elif problem_count >= 1:
        score += 5   # Minimal
    
    # 2. Technical/solution architecture (MAXIMUM 25 points)
    tech_keywords = [
        'architecture', 'system', 'platform', 'framework', 'model',
        'algorithm', 'methodology', 'approach', 'technique', 'method',
        'implementation', 'deployment', 'integration', 'scalable', 'scalability'
    ]
    tech_count = sum(1 for keyword in tech_keywords if keyword in combined_text)
    
    if tech_count >= 8:
        score += 25  # Highly technical
    elif tech_count >= 5:
        score += 18  # Technical
    elif tech_count >= 3:
        score += 10  # Some technical depth
    elif tech_count >= 1:
        score += 5   # Minimal
    
    # 3. Research/analysis depth (MAXIMUM 20 points)
    research_keywords = [
        'research', 'analysis', 'study', 'evaluation', 'assessment',
        'data', 'results', 'findings', 'evidence', 'experiment',
        'methodology', 'approach', 'investigation'
    ]
    research_count = sum(1 for keyword in research_keywords if keyword in combined_text)
    
    if research_count >= 6:
        score += 20  # Strong research
    elif research_count >= 4:
        score += 15  # Good research
    elif research_count >= 2:
        score += 8   # Some research
    elif research_count >= 1:
        score += 4   # Minimal
    
    # 4. Market/impact orientation (MAXIMUM 15 points)
    market_keywords = [
        'market', 'impact', 'benefit', 'value', 'feasibility',
        'viable', 'practical', 'application', 'real-world', 'industry',
        'user', 'customer', 'stakeholder'
    ]
    market_count = sum(1 for keyword in market_keywords if keyword in combined_text)
    
    if market_count >= 6:
        score += 15  # Strong market focus
    elif market_count >= 4:
        score += 10  # Good focus
    elif market_count >= 2:
        score += 5   # Some focus
    
    # 5. AI quality bonus (MAXIMUM 15 points) - REDUCED from 40
    ai_content_score = ai_eval.get("content_quality", {}).get("content_quality_score", 50)
    
    if ai_content_score >= 85:
        score += 15  # Excellent
    elif ai_content_score >= 75:
        score += 10  # Good
    elif ai_content_score >= 65:
        score += 5   # Adequate
    
    return max(0, min(100, score))


def _calculate_market_relevance_score(slides: List[Slide], ai_eval: Dict) -> float:
    """
    Calculate market relevance & business value score - INDUSTRY-LEVEL METRIC
    Base score: 30 (higher for good presentations)
    
    Evaluates:
    - Target market identification
    - Value proposition clarity
    - Competitive advantage
    - Revenue/business model
    - Real-world application
    """
    # Use normalized, weighted scoring to avoid saturating at 100
    score_floor = 40.0
    all_text = " ".join([s.body_text.lower() if s.body_text else "" for s in slides])
    all_titles = " ".join([s.title.lower() if s.title else "" for s in slides])
    combined_text = all_text + " " + all_titles

    # Define subcomponents with keyword sets and weights
    market_keywords = [
        'market', 'customer', 'user', 'target', 'audience',
        'segment', 'demographic', 'persona', 'stakeholder',
        'patient', 'healthcare', 'clinical', 'medical', 'hospital', 'doctor', 'provider', 'population', 'community'
    ]
    value_keywords = [
        'value', 'benefit', 'advantage', 'unique', 'innovative',
        'solve', 'efficiency', 'cost-effective', 'roi', 'savings',
        'improve', 'accuracy', 'performance', 'quality', 'reduce', 'outcome', 'impact', 'result', 'safety', 'access'
    ]
    business_keywords = [
        'revenue', 'pricing', 'monetization', 'business model',
        'subscription', 'license', 'profit', 'cost', 'investment',
        'deployment', 'adoption', 'implementation', 'solution', 'rollout', 'scaling', 'expansion', 'growth'
    ]
    competitive_keywords = [
        'competitive', 'competitor', 'alternative', 'comparison',
        'differentiation', 'advantage', 'edge', 'superior',
        'existing', 'current', 'traditional', 'conventional', 'benchmark', 'leader', 'best', 'standard'
    ]

    # Compute normalized match ratios
    def ratio(keywords):
        if not keywords:
            return 0.0
        return min(1.0, sum(1 for k in keywords if k in combined_text) / len(keywords))

    r_market = ratio(market_keywords)
    r_value = ratio(value_keywords)
    r_business = ratio(business_keywords)
    r_comp = ratio(competitive_keywords)

    # Weights (sum to 1.0)
    w_market = 0.30
    w_value = 0.35
    w_business = 0.20
    w_comp = 0.15

    normalized = (r_market * w_market) + (r_value * w_value) + (r_business * w_business) + (r_comp * w_comp)

    # Map normalized [0,1] to [score_floor, 100]
    score = score_floor + normalized * (100.0 - score_floor)
    return max(score_floor, min(100.0, score))


def _calculate_execution_feasibility_score(slides: List[Slide], ai_eval: Dict) -> float:
    """
    Calculate execution quality & feasibility score - INDUSTRY-LEVEL METRIC
    Base score: 30 (higher for good presentations)
    
    Evaluates:
    - Roadmap/timeline clarity
    - Resource planning
    - Risk assessment
    - Milestones & deliverables
    - Team/expertise mention
    """
    # Normalized weighted scoring to avoid easy 100s
    score_floor = 40.0
    all_text = " ".join([s.body_text.lower() if s.body_text else "" for s in slides])
    all_titles = " ".join([s.title.lower() if s.title else "" for s in slides])
    combined_text = all_text + " " + all_titles

    roadmap_keywords = [
        'roadmap', 'timeline', 'phase', 'milestone', 'schedule',
        'sprint', 'quarter', 'delivery', 'deadline', 'plan', 'implementation', 'rollout', 'launch', 'release', 'go-live'
    ]
    resource_keywords = [
        'resource', 'team', 'budget', 'cost', 'infrastructure',
        'technology', 'tool', 'personnel', 'expertise', 'skill', 'staff', 'capacity', 'equipment', 'facility', 'support'
    ]
    risk_keywords = [
        'risk', 'challenge', 'mitigation', 'contingency',
        'limitation', 'constraint', 'dependency', 'assumption', 'barrier', 'obstacle', 'issue', 'problem'
    ]
    implementation_keywords = [
        'implementation', 'deployment', 'integration', 'testing',
        'development', 'execution', 'rollout', 'launch', 'validation', 'pilot', 'trial', 'demo', 'prototype'
    ]

    def ratio(keywords):
        if not keywords:
            return 0.0
        return min(1.0, sum(1 for k in keywords if k in combined_text) / len(keywords))

    r_roadmap = ratio(roadmap_keywords)
    r_resource = ratio(resource_keywords)
    r_risk = ratio(risk_keywords)
    r_impl = ratio(implementation_keywords)

    # Weights
    w_roadmap = 0.30
    w_resource = 0.30
    w_risk = 0.20
    w_impl = 0.20

    normalized = (r_roadmap * w_roadmap) + (r_resource * w_resource) + (r_risk * w_risk) + (r_impl * w_impl)
    score = score_floor + normalized * (100.0 - score_floor)
    return max(score_floor, min(100.0, score))


def _calculate_data_evidence_score(slides: List[Slide], ai_eval: Dict) -> float:
    """
    Calculate data & evidence quality score - INDUSTRY-LEVEL METRIC
    Base score: 30 (higher for good presentations)
    
    Evaluates:
    - Statistics & metrics
    - Research citations
    - Case studies & examples
    - Validation & proof
    - Quantitative data
    """
    # Normalized weighted scoring for evidence and data
    score_floor = 40.0
    all_text = " ".join([s.body_text.lower() if s.body_text else "" for s in slides])
    all_titles = " ".join([s.title.lower() if s.title else "" for s in slides])
    combined_text = all_text + " " + all_titles

    stat_keywords = [
        'data', 'statistics', 'metric', 'measurement', 'kpi',
        'performance', 'result', 'outcome', 'finding', 'accuracy', 'rate', 'score', 'increase', 'decrease', 'change', 'trend'
    ]
    research_keywords = [
        'research', 'study', 'survey', 'experiment', 'test',
        'validation', 'proof', 'evidence', 'verified', 'confirmed', 'publication', 'journal', 'clinical', 'peer-reviewed'
    ]
    case_keywords = [
        'case study', 'example', 'demonstration', 'pilot',
        'prototype', 'trial', 'real-world', 'practical', 'application', 'implementation', 'success', 'failure'
    ]

    def ratio(keywords):
        if not keywords:
            return 0.0
        return min(1.0, sum(1 for k in keywords if k in combined_text) / len(keywords))

    r_stats = ratio(stat_keywords)
    r_research = ratio(research_keywords)
    r_case = ratio(case_keywords)

    # Give additional weight for presence of numeric values (counts, percentages)
    has_numbers = any(char.isdigit() for char in combined_text)
    number_bonus = 0.1 if has_numbers else 0.0

    # Weights: stats 50%, research 35%, case 15%
    normalized = (r_stats * 0.5) + (r_research * 0.35) + (r_case * 0.15)
    normalized = min(1.0, normalized + number_bonus)

    score = score_floor + normalized * (100.0 - score_floor)
    return max(score_floor, min(100.0, score))


def _calculate_impact_scalability_score(slides: List[Slide], ai_eval: Dict) -> float:
    """
    Calculate impact & scalability score - INDUSTRY-LEVEL METRIC
    Base score: 10 (LOWERED from 20 - be stricter!)
    
    Evaluates:
    - Social/environmental impact
    - Growth potential
    - Scalability plan
    - Long-term vision
    - Sustainability
    
    Separates transformative projects from incremental improvements!
    """
    score = 10.0  # Base score LOWERED from 20
    
    all_text = " ".join([s.body_text.lower() if s.body_text else "" for s in slides])
    all_titles = " ".join([s.title.lower() if s.title else "" for s in slides])
    combined_text = all_text + " " + all_titles
    
    # 1. Impact focus (MAXIMUM 35 points)
    impact_keywords = [
        'impact', 'transform', 'change', 'improve', 'benefit',
        'social', 'environmental', 'sustainability', 'community',
        'healthcare', 'education', 'society'
    ]
    impact_count = sum(1 for keyword in impact_keywords if keyword in combined_text)
    
    if impact_count >= 8:
        score += 35
    elif impact_count >= 6:
        score += 26
    elif impact_count >= 4:
        score += 18
    elif impact_count >= 2:
        score += 10
    elif impact_count >= 1:
        score += 4
    
    # 2. Scalability (MAXIMUM 30 points)
    scale_keywords = [
        'scalable', 'scalability', 'scale', 'growth', 'expand',
        'expansion', 'replicate', 'widespread', 'global'
    ]
    scale_count = sum(1 for keyword in scale_keywords if keyword in combined_text)
    
    if scale_count >= 6:
        score += 30
    elif scale_count >= 4:
        score += 22
    elif scale_count >= 2:
        score += 12
    elif scale_count >= 1:
        score += 4
    
    # 3. Vision & future (MAXIMUM 15 points)
    vision_keywords = [
        'vision', 'future', 'long-term', 'potential', 'opportunity',
        'evolution', 'roadmap', 'sustainable'
    ]
    vision_count = sum(1 for keyword in vision_keywords if keyword in combined_text)
    
    if vision_count >= 5:
        score += 15
    elif vision_count >= 3:
        score += 10
    elif vision_count >= 1:
        score += 4
    
    return max(0, min(100, score))


def _calculate_professional_quality_score(slides: List[Slide], ai_eval: Dict) -> float:
    """
    Calculate professional presentation quality score - INDUSTRY-LEVEL METRIC
    Base score: 20 (LOWERED from 30 - be stricter!)
    
    Evaluates:
    - Storytelling & narrative
    - Engagement & persuasion
    - Credibility & authority
    - Call-to-action
    - Professional polish
    
    Separates investor-ready presentations from classroom slides!
    """
    score = 20.0  # Base score LOWERED from 30
    
    all_text = " ".join([s.body_text.lower() if s.body_text else "" for s in slides])
    all_titles = " ".join([s.title.lower() if s.title else "" for s in slides])
    combined_text = all_text + " " + all_titles
    
    # 1. Storytelling & narrative (MAXIMUM 25 points)
    story_keywords = [
        'story', 'journey', 'narrative', 'experience',
        'background', 'context', 'motivation', 'why'
    ]
    story_count = sum(1 for keyword in story_keywords if keyword in combined_text)
    
    if story_count >= 5:
        score += 25
    elif story_count >= 3:
        score += 18
    elif story_count >= 1:
        score += 8
    
    # 2. Engagement & persuasion (MAXIMUM 25 points)
    engage_keywords = [
        'key', 'important', 'critical', 'essential', 'significant',
        'innovative', 'revolutionary', 'breakthrough', 'unique'
    ]
    engage_count = sum(1 for keyword in engage_keywords if keyword in combined_text)
    
    if engage_count >= 6:
        score += 25
    elif engage_count >= 4:
        score += 18
    elif engage_count >= 2:
        score += 10
    elif engage_count >= 1:
        score += 4
    
    # 3. Credibility & authority (MAXIMUM 15 points)
    cred_keywords = [
        'expert', 'proven', 'validated', 'certified', 'established',
        'recognized', 'award', 'publication', 'patent'
    ]
    cred_count = sum(1 for keyword in cred_keywords if keyword in combined_text)
    
    if cred_count >= 3:
        score += 15
    elif cred_count >= 1:
        score += 8
    
    # 4. Call-to-action (MAXIMUM 5 points)
    cta_keywords = [
        'next step', 'action', 'contact', 'join', 'invest',
        'partner', 'collaborate', 'get started', 'learn more'
    ]
    cta_count = sum(1 for keyword in cta_keywords if keyword in combined_text)
    
    if cta_count >= 2:
        score += 5
    elif cta_count >= 1:
        score += 2
    
    return max(0, min(100, score))


def _calculate_grade(score: float) -> str:
    """Convert numeric score to letter grade (Very Lenient Scale)"""
    if score >= 75:
        return "A"
    elif score >= 55:
        return "B"
    elif score >= 40:
        return "C"
    elif score >= 25:
        return "D"
    else:
        return "F"


def _identify_strengths(slide_scores: List[SlideScore], features: any, ai_eval: Dict) -> List[str]:
    """Identify presentation strengths"""
    strengths = []
    
    # Check component scores
    avg_clarity = statistics.mean([s.clarity_score for s in slide_scores])
    avg_structure = statistics.mean([s.structure_score for s in slide_scores])
    avg_blooms = statistics.mean([s.blooms_score for s in slide_scores])
    
    if avg_clarity >= 80:
        strengths.append("Excellent clarity and readability")
    
    if avg_structure >= 85:
        strengths.append("Well-organized structure with clear sections")
    
    if avg_blooms >= 70:
        strengths.append("High cognitive engagement (Bloom's taxonomy)")
    
    redundancy_score = features.get("redundancy", {}).get("redundancy_score", 0)
    if redundancy_score < 0.2:
        strengths.append("No redundant content - excellent variety")

    semantic_density = features.get("semantic_density", {}).get("overall_density", 0)
    if semantic_density >= 0.7:
        strengths.append("Rich semantic content with conceptual diversity")
    
    # Add AI-identified strengths
    ai_strengths = ai_eval.get("content_quality", {}).get("strengths", [])
    strengths.extend(ai_strengths[:2])  # Add top 2 AI strengths
    
    return strengths[:5]  # Return top 5


def _identify_weaknesses(slide_scores: List[SlideScore], features: any, ai_eval: Dict) -> List[str]:
    """Identify presentation weaknesses"""
    weaknesses = []
    
    # Check component scores
    avg_design = statistics.mean([s.design_score for s in slide_scores])
    avg_readability = statistics.mean([s.readability_score for s in slide_scores])
    
    if avg_design < 60:
        weaknesses.append("Layout and visual design needs improvement")
    
    if avg_readability < 65:
        weaknesses.append("Text readability could be enhanced")
    
    redundancy_score = features.get("redundancy", {}).get("redundancy_score", 0)
    if redundancy_score > 0.3:
        weaknesses.append("Some redundant content detected")

    avg_layout_quality = features.get("layout_quality", {}).get("overall_quality", 100)
    if avg_layout_quality < 60:
        weaknesses.append("Inconsistent layout quality across slides")
    
    # Add AI-identified weaknesses
    ai_weaknesses = ai_eval.get("content_quality", {}).get("weaknesses", [])
    weaknesses.extend(ai_weaknesses[:2])  # Add top 2 AI weaknesses
    
    return weaknesses[:5]  # Return top 5
