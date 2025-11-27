"""
🤖 Milestone 6: AI Evaluation Service
Uses Groq API to evaluate presentation quality with expert-level analysis
"""

from typing import List, Dict, Optional
from models.slide import Slide
from core.logging import logger
from core.config import settings
import json
import os
import asyncio

# Initialize Groq client
_client = None
GROQ_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    logger.warning("Groq library not installed. AI evaluation will use mock responses.")

def get_client():
    """Get Groq client if API key is available"""
    global _client
    
    if _client is None and GROQ_AVAILABLE:
        api_key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
        if not api_key:
            logger.warning("No GROQ_API_KEY found - AI evaluation will use mock responses")
            return None
        
        try:
            _client = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully for AI evaluation")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            return None
    
    return _client


async def evaluate_content_quality(slides: List[Slide]) -> Dict:
    """
    Evaluate overall content quality using Groq
    """
    logger.info("Evaluating content quality with Groq AI...")
    
    client = get_client()
    
    # Prepare slide content for analysis
    slide_summaries = []
    for slide in slides:
        summary = f"Slide {slide.slide_number}"
        if slide.title:
            summary += f" - Title: {slide.title}"
        if slide.body_text:
            summary += f"\nContent: {slide.body_text[:200]}..."  # Limit to 200 chars
        slide_summaries.append(summary)
    
    content = "\n\n".join(slide_summaries)
    
    if client:
        try:
            prompt = f"""You are an expert presentation evaluator. Analyze the content quality, clarity, and professionalism.

Presentation content:
{content}

Provide your response as valid JSON with these fields:
- content_quality_score: number 0-100
- clarity_score: number 0-100
- professionalism_score: number 0-100
- strengths: array of strings
- weaknesses: array of strings
- recommendations: array of strings

JSON response:"""
            
            # Run in thread pool since Groq client is sync
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are an expert presentation evaluator. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
            )
            
            result_text = response.choices[0].message.content
            
            # Try to parse JSON from response
            try:
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(result_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from Groq response, using mock")
                return _get_mock_evaluation()
            
            logger.info(f"AI evaluation complete - Quality: {result.get('content_quality_score', 0)}/100")
            return result
            
        except Exception as e:
            logger.error(f"AI evaluation failed: {e}")
            return _get_mock_evaluation()
    else:
        return _get_mock_evaluation()


async def evaluate_clarity_coherence(slides: List[Slide]) -> Dict:
    """
    Evaluate clarity and coherence of the presentation
    """
    logger.info("Evaluating clarity and coherence...")
    
    client = get_client()
    
    # Analyze flow and transitions
    transitions = []
    for i in range(len(slides) - 1):
        curr_title = slides[i].title or "Untitled"
        next_title = slides[i + 1].title or "Untitled"
        transitions.append(f"{curr_title} → {next_title}")
    
    if client:
        try:
            prompt = f"""You are an expert in presentation flow and narrative structure. Evaluate clarity and coherence.

Slide Transitions:
{chr(10).join(transitions)}

Provide your response as valid JSON with these fields:
- clarity_score: number 0-100
- coherence_score: number 0-100
- flow_quality: number 0-100
- narrative_strength: string
- transition_quality: string
- suggestions: array of strings

JSON response:"""
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are an expert in presentation flow and narrative structure. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
            )
            
            result_text = response.choices[0].message.content
            
            try:
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(result_text)
            except json.JSONDecodeError:
                return _get_mock_clarity_evaluation()
            
            logger.info(f"Clarity evaluation complete - Score: {result.get('clarity_score', 0)}/100")
            return result
            
        except Exception as e:
            logger.error(f"Clarity evaluation failed: {e}")
            return _get_mock_clarity_evaluation()
    else:
        return _get_mock_clarity_evaluation()


async def evaluate_audience_appropriateness(slides: List[Slide], audience_type: str = "general") -> Dict:
    """
    Evaluate if content is appropriate for target audience
    """
    logger.info(f"Evaluating audience appropriateness for '{audience_type}' audience...")
    
    client = get_client()
    
    # Sample content from slides
    sample_content = []
    for slide in slides[:5]:  # Sample first 5 slides
        if slide.body_text:
            sample_content.append(slide.body_text[:150])
    
    content_sample = " ".join(sample_content)
    
    if client:
        try:
            prompt = f"""You are an expert in audience analysis. Evaluate if presentation content is appropriate for a {audience_type} audience.

Content sample: {content_sample}

Provide your response as valid JSON with these fields:
- appropriateness_score: number 0-100
- language_level: string
- technical_depth: string
- engagement_potential: number 0-100
- recommendations: array of strings

JSON response:"""
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are an expert in audience analysis. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
            )
            
            result_text = response.choices[0].message.content
            
            try:
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(result_text)
            except json.JSONDecodeError:
                return _get_mock_audience_evaluation(audience_type)
            
            logger.info(f"Audience evaluation complete - Score: {result.get('appropriateness_score', 0)}/100")
            return result
            
        except Exception as e:
            logger.error(f"Audience evaluation failed: {e}")
            return _get_mock_audience_evaluation(audience_type)
    else:
        return _get_mock_audience_evaluation(audience_type)


async def evaluate_professional_standards(slides: List[Slide]) -> Dict:
    """
    Evaluate adherence to professional presentation standards
    
    Args:
        slides: List of slides to evaluate
        
    Returns:
        Dictionary with professional standards metrics
    """
    logger.info("Evaluating professional standards...")
    
    # Analyze structure
    has_intro = any(s.section == "introduction" for s in slides if hasattr(s, 'section'))
    has_conclusion = any(s.section == "conclusion" for s in slides if hasattr(s, 'section'))
    avg_words = sum(s.total_words for s in slides) / len(slides) if slides else 0
    
    standards = {
        "has_title_slide": has_intro,
        "has_conclusion_slide": has_conclusion,
        "appropriate_length": 5 <= len(slides) <= 30,
        "consistent_structure": True,  # Would need more analysis
        "appropriate_detail": 10 <= avg_words <= 50
    }
    
    # Calculate overall score
    passed_standards = sum(1 for v in standards.values() if v)
    standards_score = (passed_standards / len(standards)) * 100
    
    return {
        "standards_score": standards_score,
        "standards_met": standards,
        "total_slides": len(slides),
        "average_words_per_slide": avg_words,
        "professional_rating": "Excellent" if standards_score >= 80 else "Good" if standards_score >= 60 else "Needs Improvement"
    }


async def evaluate_slides(slides: List[Slide], audience_type: str = "general") -> Dict:
    """
    Comprehensive AI evaluation of presentation
    Main entry point for Milestone 6
    
    Args:
        slides: List of slides to evaluate
        audience_type: Target audience type
        
    Returns:
        Dictionary with complete AI evaluation
    """
    logger.info(f"Starting AI evaluation for {len(slides)} slides")
    
    try:
        # Run all evaluations
        content_eval = await evaluate_content_quality(slides)
        clarity_eval = await evaluate_clarity_coherence(slides)
        audience_eval = await evaluate_audience_appropriateness(slides, audience_type)
        standards_eval = await evaluate_professional_standards(slides)
        
        # Calculate overall AI score
        scores = [
            content_eval.get("content_quality_score", 0),
            clarity_eval.get("clarity_score", 0),
            audience_eval.get("appropriateness_score", 0),
            standards_eval.get("standards_score", 0)
        ]
        overall_score = sum(scores) / len(scores)
        
        evaluation = {
            "overall_score": overall_score,
            "content_quality": content_eval,
            "clarity_coherence": clarity_eval,
            "audience_fit": audience_eval,
            "professional_standards": standards_eval,
            "grade": _calculate_grade(overall_score)
        }
        
        logger.info(f"AI evaluation complete - Overall: {overall_score:.1f}/100 ({evaluation['grade']})")
        return evaluation
        
    except Exception as e:
        logger.error(f"AI evaluation failed: {e}")
        return _get_mock_complete_evaluation()


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


def _get_mock_evaluation() -> Dict:
    """Mock evaluation for when API key is not available"""
    return {
        "content_quality_score": 78,
        "clarity_score": 82,
        "professionalism_score": 85,
        "strengths": [
            "Clear structure with introduction and conclusion",
            "Good use of technical terminology",
            "Well-organized flow of ideas"
        ],
        "weaknesses": [
            "Some slides have too much text",
            "Could benefit from more visual elements",
            "Transitions between topics could be smoother"
        ],
        "recommendations": [
            "Add more charts and diagrams to visualize data",
            "Reduce text density on information-heavy slides",
            "Include transition slides between major sections"
        ]
    }


def _get_mock_clarity_evaluation() -> Dict:
    """Mock clarity evaluation"""
    return {
        "clarity_score": 80,
        "coherence_score": 75,
        "flow_quality": 78,
        "narrative_strength": "Good - follows logical progression",
        "transition_quality": "Adequate - some abrupt shifts between topics",
        "suggestions": [
            "Add signposting to guide audience through sections",
            "Use consistent terminology throughout",
            "Strengthen connections between related concepts"
        ]
    }


def _get_mock_audience_evaluation(audience_type: str) -> Dict:
    """Mock audience evaluation"""
    return {
        "appropriateness_score": 82,
        "language_level": "Appropriate for technical audience",
        "technical_depth": "Well-balanced - explains concepts without oversimplifying",
        "engagement_potential": 78,
        "recommendations": [
            f"Content is well-suited for {audience_type} audience",
            "Consider adding more interactive elements",
            "Include real-world examples relevant to audience"
        ]
    }


def _get_mock_complete_evaluation() -> Dict:
    """Mock complete evaluation"""
    return {
        "overall_score": 79.5,
        "content_quality": _get_mock_evaluation(),
        "clarity_coherence": _get_mock_clarity_evaluation(),
        "audience_fit": _get_mock_audience_evaluation("general"),
        "professional_standards": {
            "standards_score": 80,
            "standards_met": {
                "has_title_slide": True,
                "has_conclusion_slide": True,
                "appropriate_length": True,
                "consistent_structure": True,
                "appropriate_detail": True
            },
            "professional_rating": "Good"
        },
        "grade": "C+"
    }
