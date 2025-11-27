"""
🎯 Milestone 3 Evaluation Service - Smart Fallback Version
Tries OpenAI first, automatically falls back to Groq if quota/rate limit errors
"""

from typing import List, Dict, Optional
from models.slide import Slide
from core.logging import logger
from core.config import settings
import os
import asyncio

# Import OpenAI service
from services.milestone3_evaluation_service import (
    evaluate_milestone3 as evaluate_milestone3_openai,
    MILESTONE3_QUESTIONS,
    _get_mock_evaluation,
    _calculate_category_score,
    _calculate_grade,
    _extract_presentation_content,
    evaluate_single_question as evaluate_single_question_openai
)

# Import Groq for fallback
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq library not available - install with: pip install groq")


async def evaluate_single_question_groq(question_data: Dict, presentation_content: str, client) -> Dict:
    """Evaluate a single question using Groq API"""
    question_name = question_data["name"]
    question_text = question_data["question"]
    rating_scale = question_data["rating_scale"]
    
    logger.info(f"Evaluating with Groq: {question_name}")
    
    # Build prompt
    scale_text = "\n".join([f"- {scale}" for scale in rating_scale])
    
    evaluation_prompt = f"""You are an expert venture assessment evaluator. Evaluate the presentation based on this specific criterion:

**Criterion: {question_name}**
{question_text}

**Available Rating Levels (choose ONE that best matches):**
{scale_text}

**Presentation Content:**
{presentation_content[:4000]}

**Instructions:**
1. Analyze the presentation content for this specific criterion
2. Select the EXACT rating level from the list above that best matches the presentation
3. Provide a brief justification (2-3 sentences) explaining why this rating was chosen
4. Respond in this EXACT format:
   RATING: [exact text from rating scale]
   JUSTIFICATION: [your explanation]

Your response:"""

    # Use Groq API
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Faster model
        messages=[
            {"role": "system", "content": "You are an expert venture assessment evaluator."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.3,
        max_tokens=512
    )
    
    result_text = response.choices[0].message.content.strip()
    
    # Parse response
    import re
    rating_match = re.search(r'RATING:\s*(.+)', result_text, re.IGNORECASE)
    justification_match = re.search(r'JUSTIFICATION:\s*(.+)', result_text, re.IGNORECASE | re.DOTALL)
    
    rating_text = rating_match.group(1).strip() if rating_match else result_text.split('\n')[0]
    justification = justification_match.group(1).strip() if justification_match else "No justification provided"
    
    # Map to rating scale
    from services.milestone3_evaluation_service import _map_rating_to_scale
    rating_info = _map_rating_to_scale(rating_text, rating_scale)
    
    # Get improvement suggestion from Groq
    improvement_question = question_data.get("improvement_question", "")
    if improvement_question:
        improvement_prompt = f"""Based on the evaluation above for {question_name}, answer this question:

{improvement_question}

Presentation Content:
{presentation_content[:4000]}

Provide a specific, actionable improvement suggestion (1-2 sentences)."""

        improvement_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert venture assessment coach providing actionable feedback."},
                {"role": "user", "content": improvement_prompt}
            ],
            temperature=0.3,
            max_tokens=256
        )
        improvement_suggestion = improvement_response.choices[0].message.content.strip()
    else:
        improvement_suggestion = "No specific improvement suggestion available."
    
    return {
        "question_id": question_data["id"],
        "question_name": question_name,
        "rating_level": rating_info["rating_level"],
        "rating_index": rating_info["rating_index"],
        "score": rating_info["score"],
        "justification": justification,
        "improvement_suggestion": improvement_suggestion,
        "token_usage": {"question_total": 0}
    }


async def evaluate_milestone3_groq_impl(slides: List[Slide]) -> Dict:
    """Evaluate using Groq API"""
    groq_key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
    
    if not groq_key:
        raise Exception("No Groq API key found")
    
    client = Groq(api_key=groq_key)
    presentation_content = _extract_presentation_content(slides)
    
    results = []
    total_score = 0
    
    for idx, question_data in enumerate(MILESTONE3_QUESTIONS, 1):
        logger.info(f"  [{idx}/18] Evaluating: {question_data['name']}...")
        result = await evaluate_single_question_groq(question_data, presentation_content, client)
        results.append(result)
        total_score += result["score"]
        logger.info(f"    ✓ {result['rating_level']} ({result['score']:.1f}/100)")
        
        if idx < len(MILESTONE3_QUESTIONS):
            await asyncio.sleep(0.2)  # Reduced delay - Groq has generous rate limits
    
    overall_score = total_score / len(MILESTONE3_QUESTIONS) if results else 0
    
    category_scores = {
        "Problem & Solution": _calculate_category_score(results, [1, 4, 5, 6]),
        "Market Analysis": _calculate_category_score(results, [2, 3, 7, 8, 9]),
        "Business Model": _calculate_category_score(results, [10, 11]),
        "Go-To-Market": _calculate_category_score(results, [12, 13]),
        "Execution": _calculate_category_score(results, [14, 15, 16]),
        "Presentation": _calculate_category_score(results, [17, 18])
    }
    
    return {
        "overall_score": overall_score,
        "total_questions": len(MILESTONE3_QUESTIONS),
        "evaluations": results,
        "category_scores": category_scores,
        "grade": _calculate_grade(overall_score),
        "token_usage": {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "average_per_question": 0,
            "estimated_cost_usd": 0.0,
            "provider": "Groq (FREE)"
        }
    }


async def evaluate_milestone3(slides: List[Slide]) -> Dict:
    """
    Milestone 3 Evaluation - Groq ONLY (no OpenAI fallback)
    
    OpenAI has no credits, so we skip it entirely to avoid retry delays.
    """
    groq_key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
    
    # Use Groq ONLY (no OpenAI fallback to avoid retry delays)
    if groq_key and GROQ_AVAILABLE:
        logger.info("🟢 Using Groq API (FREE, fast, no limits)...")
        result = await evaluate_milestone3_groq_impl(slides)
        logger.info("✅ Groq evaluation successful!")
        return result
    
    # No Groq? Use mock data (no OpenAI to avoid retries)
    logger.warning("⚠️ Groq not available - using mock evaluation")
    
    results = []
    total_score = 0
    
    for question_data in MILESTONE3_QUESTIONS:
        result = _get_mock_evaluation(question_data)
        results.append(result)
        total_score += result["score"]
    
    overall_score = total_score / len(MILESTONE3_QUESTIONS) if results else 0
    
    category_scores = {
        "Problem & Solution": _calculate_category_score(results, [1, 4, 5, 6]),
        "Market Analysis": _calculate_category_score(results, [2, 3, 7, 8, 9]),
        "Business Model": _calculate_category_score(results, [10, 11]),
        "Go-To-Market": _calculate_category_score(results, [12, 13]),
        "Execution": _calculate_category_score(results, [14, 15, 16]),
        "Presentation": _calculate_category_score(results, [17, 18])
    }
    
    return {
        "overall_score": overall_score,
        "total_questions": len(MILESTONE3_QUESTIONS),
        "evaluations": results,
        "category_scores": category_scores,
        "grade": _calculate_grade(overall_score),
        "token_usage": {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "average_per_question": 0,
            "estimated_cost_usd": 0.0,
            "provider": "Mock (no Groq key)"
        }
    }
