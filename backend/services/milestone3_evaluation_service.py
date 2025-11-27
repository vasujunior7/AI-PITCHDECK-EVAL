"""
🎯 Milestone 3: Venture Assessment Evaluation Service
Evaluates presentations based on 18 Venture Assessment criteria with specific grading scales
Uses OpenAI API for Milestone 3 evaluation
"""

from typing import List, Dict, Optional
from models.slide import Slide
from core.logging import logger
from core.config import settings
from openai import OpenAI, RateLimitError
import json
import re
import asyncio
from utils.rate_limit_handler import handle_rate_limit_with_backoff_async

# Initialize OpenAI client
_client = None

def get_client():
    """Get OpenAI client for Milestone 3 evaluation"""
    global _client
    
    if _client is None:
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            logger.warning("No OPENAI_API_KEY found - Milestone 3 evaluation will use mock responses")
            return None
        
        try:
            # Configure client with better retry settings for rate limits
            _client = OpenAI(
                api_key=api_key,
                max_retries=5,  # Increase retries for rate limits
                timeout=60.0   # 60 second timeout
            )
            logger.info(f"OpenAI client initialized for Milestone 3 evaluation (max_retries=5)")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return None
    
    return _client


# Define all 18 evaluation questions with their specific grading scales
MILESTONE3_QUESTIONS = [
    {
        "id": 1,
        "name": "Problem/Opportunity Statement",
        "question": "Evaluate the Problem/Opportunity Statement in the presentation.",
        "rating_scale": [
            "(missing)",
            "(ambiguous)",
            "(basic understanding)",
            "(well-defined)",
            "(highly specific)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Problem/Opportunity Statement?"
    },
    {
        "id": 2,
        "name": "Market Segmentation",
        "question": "Evaluate the Market Segmentation research and analysis in the presentation.",
        "rating_scale": [
            "(missing)",
            "(limited)",
            "(basic)",
            "(thorough)",
            "(exceptional)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Market Segmentation?"
    },
    {
        "id": 3,
        "name": "Customer Persona",
        "question": "Evaluate the Customer Persona in the presentation.",
        "rating_scale": [
            "(poor)",
            "(basic)",
            "(adequate)",
            "(good)",
            "(excellent)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Customer Persona?"
    },
    {
        "id": 4,
        "name": "Problem Validation",
        "question": "Evaluate the Problem Validation provided in the presentation.",
        "rating_scale": [
            "(missing)",
            "(limited)",
            "(incomplete)",
            "(comprehensive)",
            "(extensive & convincing)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Problem Validation?"
    },
    {
        "id": 5,
        "name": "Solution Statement",
        "question": "Evaluate the Solution Statement in the presentation.",
        "rating_scale": [
            "(incomprehensible)",
            "(partially coherent)",
            "(basic clarity)",
            "(effective)",
            "(innovative & effective)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Solution Statement?"
    },
    {
        "id": 6,
        "name": "Solution Validation",
        "question": "Evaluate the Solution Validation in the presentation.",
        "rating_scale": [
            "(not validated)",
            "(minimal validation)",
            "(limited validation)",
            "(sufficient validation)",
            "(comprehensive validation)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Solution Validation?"
    },
    {
        "id": 7,
        "name": "Competition Analysis",
        "question": "Evaluate the Competition Analysis in the presentation.",
        "rating_scale": [
            "(missing)",
            "(limited)",
            "(adequate)",
            "(thorough)",
            "(comprehensive)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Competition Analysis?"
    },
    {
        "id": 8,
        "name": "Unique Value Proposition",
        "question": "Evaluate the Unique Value Proposition in the presentation.",
        "rating_scale": [
            "(undefined)",
            "(unclear solution)",
            "(basic solution)",
            "(differentiated solution)",
            "(clear differentiation)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Unique Value Proposition?"
    },
    {
        "id": 9,
        "name": "Market Size",
        "question": "Evaluate the Market Size estimation in the presentation.",
        "rating_scale": [
            "(missing)",
            "(limited)",
            "(basic)",
            "(well-supported)",
            "(thorough & referenced)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Market Size?"
    },
    {
        "id": 10,
        "name": "Revenue/Pricing Models",
        "question": "Evaluate the Revenue/Pricing Models in the presentation.",
        "rating_scale": [
            "(inadequate)",
            "(basic)",
            "(adequate)",
            "(detailed)",
            "(comprehensive)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Revenue/Pricing Models?"
    },
    {
        "id": 11,
        "name": "Lean Canvas",
        "question": "Evaluate the Lean Canvas in the presentation.",
        "rating_scale": [
            "(missing)",
            "(limited)",
            "(adequate)",
            "(strong)",
            "(exceptional)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Lean Canvas?"
    },
    {
        "id": 12,
        "name": "GTM Approach",
        "question": "Evaluate the GTM (Go-To-Market) Approach in the presentation.",
        "rating_scale": [
            "(not defined)",
            "(basic)",
            "(clearly defined)",
            "(well-defined)",
            "(comprehensive)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on GTM Approach?"
    },
    {
        "id": 13,
        "name": "Sales Strategy",
        "question": "Evaluate the Sales Strategy in the presentation.",
        "rating_scale": [
            "(absent)",
            "(basic)",
            "(structured)",
            "(well-developed)",
            "(comprehensive & aligned)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Sales Strategy?"
    },
    {
        "id": 14,
        "name": "Finance",
        "question": "Evaluate the Financial plan in the presentation.",
        "rating_scale": [
            "(inadequate)",
            "(incomplete)",
            "(basic)",
            "(comprehensive but flawed)",
            "(thorough & aligned)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Finance?"
    },
    {
        "id": 15,
        "name": "Milestone Planning",
        "question": "Evaluate the Milestone Planning in the presentation.",
        "rating_scale": [
            "(poor)",
            "(basic)",
            "(adequate)",
            "(detailed)",
            "(comprehensive)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Milestone Planning?"
    },
    {
        "id": 16,
        "name": "Skills & Team Alignment",
        "question": "Evaluate the Skills & Team Alignment in the presentation.",
        "rating_scale": [
            "(unprepared)",
            "(basic awareness)",
            "(adequate planning)",
            "(detailed understanding)",
            "(perfect alignment)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Skills & Team Alignment?"
    },
    {
        "id": 17,
        "name": "Presentation Skills",
        "question": "Evaluate the Presentation Skills demonstrated in the presentation.",
        "rating_scale": [
            "(Poor)",
            "(Basic)",
            "(Adequate)",
            "(Strong)",
            "(Exceptional)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Presentation Skills?"
    },
    {
        "id": 18,
        "name": "Engagement & Interaction",
        "question": "Evaluate the Engagement & Interaction in the presentation.",
        "rating_scale": [
            "(Poor)",
            "(Basic)",
            "(Adequate)",
            "(Strong)",
            "(Exceptional)"
        ],
        "improvement_question": "What is the one thing that the students need to do to improve their output on Engagement & Interaction?"
    }
]


def _extract_presentation_content(slides: List[Slide]) -> str:
    """Extract all text content from slides for context"""
    content_parts = []
    
    for slide in slides:
        slide_content = f"Slide {slide.slide_number}"
        if slide.title:
            slide_content += f"\nTitle: {slide.title}"
        if slide.body_text:
            # Limit body text to avoid token limits
            body = slide.body_text[:500] if len(slide.body_text) > 500 else slide.body_text
            slide_content += f"\nContent: {body}"
        content_parts.append(slide_content)
    
    return "\n\n".join(content_parts)


def _map_rating_to_scale(rating_text: str, rating_scale: List[str]) -> Optional[Dict]:
    """
    Map AI response to the correct rating level from the scale
    
    Returns:
        Dict with 'rating_level', 'rating_index' (0-4), and 'score' (0-100)
    """
    rating_text_lower = rating_text.lower().strip()
    
    # Try exact match first
    for idx, scale_item in enumerate(rating_scale):
        scale_lower = scale_item.lower().strip().replace("(", "").replace(")", "")
        if scale_lower in rating_text_lower or rating_text_lower in scale_lower:
            # Use lenient scoring: 0-4 maps to 40-100 range (40, 55, 70, 85, 100)
            # This prevents harsh penalties (0 or 25) for lower ratings
            min_score = 40
            max_idx = len(rating_scale) - 1
            score = min_score + (idx / max_idx) * (100 - min_score)
            return {
                "rating_level": scale_item,
                "rating_index": idx,
                "score": score
            }
    
    # Try keyword matching
    keywords_map = {
        0: ["missing", "not", "absent", "inadequate", "poor", "undefined", "incomprehensible", "unprepared"],
        1: ["limited", "basic", "ambiguous", "minimal", "incomplete", "unclear", "inadequate"],
        2: ["adequate", "some", "partial", "basic", "structured", "incomplete"],
        3: ["well", "thorough", "good", "effective", "comprehensive", "detailed", "strong", "clear"],
        4: ["highly", "exceptional", "extensive", "innovative", "comprehensive", "perfect", "excellent", "convincing"]
    }
    
    for idx in range(len(rating_scale)):
        for keyword in keywords_map.get(idx, []):
            if keyword in rating_text_lower:
                min_score = 40
                max_idx = len(rating_scale) - 1
                score = min_score + (idx / max_idx) * (100 - min_score)
                return {
                    "rating_level": rating_scale[idx],
                    "rating_index": idx,
                    "score": score
                }
    
    # Default to middle if no match
    logger.warning(f"Could not map rating '{rating_text}' to scale, defaulting to middle")
    default_idx = len(rating_scale) // 2
    min_score = 40
    max_idx = len(rating_scale) - 1
    score = min_score + (default_idx / max_idx) * (100 - min_score)
    return {
        "rating_level": rating_scale[default_idx],
        "rating_index": default_idx,
        "score": score
    }


async def evaluate_single_question(
    question_data: Dict,
    presentation_content: str,
    client: Optional[OpenAI] = None
) -> Dict:
    """
    Evaluate a single Milestone 3 question
    
    Args:
        question_data: Question definition with rating scale
        presentation_content: Extracted content from presentation
        client: Optional Groq client (will get if None)
    
    Returns:
        Dict with evaluation results
    """
    if client is None:
        client = get_client()
    
    question_name = question_data["name"]
    question_text = question_data["question"]
    rating_scale = question_data["rating_scale"]
    improvement_question = question_data["improvement_question"]
    
    logger.info(f"Evaluating Milestone 3 Question {question_data['id']}: {question_name}")
    
    if client:
        try:
            # Build prompt for evaluation
            scale_text = "\n".join([f"- {scale}" for scale in rating_scale])
            
            evaluation_prompt = f"""You are an expert venture assessment evaluator. Evaluate the presentation based on this specific criterion:

**Criterion: {question_name}**
{question_text}

**Available Rating Levels (choose ONE that best matches):**
{scale_text}

**Presentation Content:**
{presentation_content[:4000]}  # Limit content to avoid token limits

**Instructions:**
1. Analyze the presentation content for this specific criterion
2. Select the EXACT rating level from the list above that best matches the presentation
3. Provide a brief justification (2-3 sentences) explaining why this rating was chosen
4. Respond in this EXACT format:
   RATING: [exact text from rating scale]
   JUSTIFICATION: [your explanation]

Your response:"""

            # Use rate limit handler with retry logic
            async def make_evaluation_request():
                return client.chat.completions.create(
                    model="gpt-4o-mini",  # Use OpenAI GPT-4o-mini for Milestone 3 evaluation (cost-effective)
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert venture assessment evaluator. You must respond with the exact format requested, selecting one rating level from the provided scale."
                        },
                        {
                            "role": "user",
                            "content": evaluation_prompt
                        }
                    ],
                    temperature=0.3,  # Lower temperature for more consistent ratings
                    max_tokens=512
                )
            
            response = await handle_rate_limit_with_backoff_async(
                make_evaluation_request,
                max_retries=5,
                initial_delay=2.0,  # Start with 2 seconds
                max_delay=60.0
            )
            
            # Track token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
            output_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
            total_tokens = usage.total_tokens if hasattr(usage, 'total_tokens') else 0
            logger.info(f"Question {question_data['id']} Evaluation - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens} tokens")
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse response
            rating_match = re.search(r'RATING:\s*(.+)', result_text, re.IGNORECASE)
            justification_match = re.search(r'JUSTIFICATION:\s*(.+)', result_text, re.IGNORECASE | re.DOTALL)
            
            rating_text = rating_match.group(1).strip() if rating_match else result_text.split('\n')[0]
            justification = justification_match.group(1).strip() if justification_match else "No justification provided"
            
            # Map to rating scale
            rating_info = _map_rating_to_scale(rating_text, rating_scale)
            
            # Get improvement suggestion
            improvement_prompt = f"""Based on your evaluation, answer this question:

{improvement_question}

**Current Rating:** {rating_info['rating_level']}
**Justification:** {justification}

Provide ONE specific, actionable improvement suggestion (2-3 sentences maximum).

Your response:"""

            # Use rate limit handler for improvement request
            async def make_improvement_request():
                return client.chat.completions.create(
                    model="gpt-4o-mini",  # Use OpenAI GPT-4o-mini for Milestone 3 evaluation (cost-effective)
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert venture assessment evaluator providing constructive feedback."
                        },
                        {
                            "role": "user",
                            "content": improvement_prompt
                        }
                    ],
                    temperature=0.5,
                    max_tokens=256
                )
            
            improvement_response = await handle_rate_limit_with_backoff_async(
                make_improvement_request,
                max_retries=5,
                initial_delay=2.0,
                max_delay=60.0
            )
            
            # Track token usage for improvement
            improvement_usage = improvement_response.usage
            improvement_input = improvement_usage.prompt_tokens if hasattr(improvement_usage, 'prompt_tokens') else 0
            improvement_output = improvement_usage.completion_tokens if hasattr(improvement_usage, 'completion_tokens') else 0
            improvement_total = improvement_usage.total_tokens if hasattr(improvement_usage, 'total_tokens') else 0
            logger.info(f"Question {question_data['id']} Improvement - Input: {improvement_input}, Output: {improvement_output}, Total: {improvement_total} tokens")
            
            improvement_suggestion = improvement_response.choices[0].message.content.strip()
            
            # Return token usage info
            question_tokens = {
                "evaluation": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total_tokens
                },
                "improvement": {
                    "input": improvement_input,
                    "output": improvement_output,
                    "total": improvement_total
                },
                "question_total": total_tokens + improvement_total
            }
            
            return {
                "question_id": question_data["id"],
                "question_name": question_name,
                "rating_level": rating_info["rating_level"],
                "rating_index": rating_info["rating_index"],
                "score": rating_info["score"],
                "justification": justification,
                "improvement_suggestion": improvement_suggestion,
                "token_usage": question_tokens
            }
            
        except Exception as e:
            logger.error(f"Error evaluating question {question_data['id']}: {e}")
            return _get_mock_evaluation(question_data)
    else:
        return _get_mock_evaluation(question_data)


async def evaluate_milestone3(slides: List[Slide]) -> Dict:
    """
    Evaluate presentation on all 18 Milestone 3 criteria
    
    Args:
        slides: List of parsed slides
    
    Returns:
        Dict with complete Milestone 3 evaluation results
    """
    logger.info(f"Starting Milestone 3 evaluation for {len(slides)} slides")
    
    client = get_client()
    presentation_content = _extract_presentation_content(slides)
    
    # Estimate input tokens (rough calculation: ~4 chars per token)
    estimated_input_tokens = len(presentation_content) // 4
    logger.info(f"Estimated presentation content tokens: ~{estimated_input_tokens}")
    
    results = []
    total_score = 0
    
    # Token usage tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens_used = 0
    
    # Evaluate each question with progress logging
    for idx, question_data in enumerate(MILESTONE3_QUESTIONS, 1):
        logger.info(f"  [{idx}/18] Evaluating: {question_data['name']}...")
        result = await evaluate_single_question(question_data, presentation_content, client)
        results.append(result)
        total_score += result["score"]
        
        # Accumulate token usage
        token_usage = result.get("token_usage", {})
        if token_usage:
            eval_tokens = token_usage.get("evaluation", {})
            improve_tokens = token_usage.get("improvement", {})
            total_input_tokens += eval_tokens.get("input", 0) + improve_tokens.get("input", 0)
            total_output_tokens += eval_tokens.get("output", 0) + improve_tokens.get("output", 0)
            total_tokens_used += token_usage.get("question_total", 0)
        
        logger.info(f"    ✓ {result['rating_level']} ({result['score']:.1f}/100)")
        
        # Add delay to avoid rate limits (except for last question)
        # OpenAI free tier: ~3 requests per minute, paid tier: much higher
        if idx < len(MILESTONE3_QUESTIONS):
            await asyncio.sleep(1.0)  # 1 second delay between questions to respect rate limits
    
    # Calculate overall average
    overall_score = total_score / len(MILESTONE3_QUESTIONS) if results else 0
    
    # Log token usage summary
    logger.info("="*60)
    logger.info("TOKEN USAGE SUMMARY - Milestone 3 Evaluation")
    logger.info("="*60)
    logger.info(f"Total Input Tokens:  {total_input_tokens:,}")
    logger.info(f"Total Output Tokens: {total_output_tokens:,}")
    logger.info(f"Total Tokens Used:   {total_tokens_used:,}")
    logger.info(f"Average per Question: {total_tokens_used / len(MILESTONE3_QUESTIONS):.0f} tokens")
    logger.info("="*60)
    
    # Calculate category scores (grouping questions)
    category_scores = {
        "Problem & Solution": _calculate_category_score(results, [1, 4, 5, 6]),  # Problem, Validation, Solution, Solution Validation
        "Market Analysis": _calculate_category_score(results, [2, 3, 7, 8, 9]),  # Segmentation, Persona, Competition, UVP, Market Size
        "Business Model": _calculate_category_score(results, [10, 11]),  # Revenue/Pricing, Lean Canvas
        "Go-To-Market": _calculate_category_score(results, [12, 13]),  # GTM, Sales Strategy
        "Execution": _calculate_category_score(results, [14, 15, 16]),  # Finance, Milestone Planning, Skills
        "Presentation": _calculate_category_score(results, [17, 18])  # Presentation Skills, Engagement
    }
    
    evaluation = {
        "overall_score": overall_score,
        "total_questions": len(MILESTONE3_QUESTIONS),
        "evaluations": results,
        "category_scores": category_scores,
        "grade": _calculate_grade(overall_score),
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens_used,
            "average_per_question": total_tokens_used / len(MILESTONE3_QUESTIONS) if results else 0,
            "estimated_cost_usd": _estimate_cost(total_tokens_used)  # Rough estimate
        }
    }
    
    logger.info(f"Milestone 3 evaluation complete - Overall: {overall_score:.1f}/100 ({evaluation['grade']})")
    logger.info(f"Estimated Cost: ${evaluation['token_usage']['estimated_cost_usd']:.4f} USD")
    return evaluation


def _estimate_cost(total_tokens: int) -> float:
    """
    Estimate cost based on OpenAI pricing for gpt-4o-mini
    Pricing (as of 2024): 
    - Input: $0.15 per 1M tokens
    - Output: $0.60 per 1M tokens
    - Average: ~$0.375 per 1M tokens (assuming 80% input, 20% output)
    This is a rough estimate - actual pricing may vary
    """
    # Rough estimate: weighted average (most tokens are input)
    input_cost = (total_tokens * 0.8 / 1_000_000) * 0.15  # 80% input at $0.15/1M
    output_cost = (total_tokens * 0.2 / 1_000_000) * 0.60   # 20% output at $0.60/1M
    return input_cost + output_cost


def _calculate_category_score(results: List[Dict], question_ids: List[int]) -> float:
    """Calculate average score for a category of questions"""
    category_results = [r for r in results if r["question_id"] in question_ids]
    if not category_results:
        return 0.0
    return sum(r["score"] for r in category_results) / len(category_results)


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


def _get_mock_evaluation(question_data: Dict) -> Dict:
    """Mock evaluation for when API key is not available"""
    rating_scale = question_data["rating_scale"]
    default_idx = len(rating_scale) // 2  # Middle rating
    
    return {
        "question_id": question_data["id"],
        "question_name": question_data["name"],
        "rating_level": rating_scale[default_idx],
        "rating_index": default_idx,
        "score": (default_idx / (len(rating_scale) - 1)) * 100,
        "justification": f"Mock evaluation for {question_data['name']} - default rating assigned.",
        "improvement_suggestion": f"To improve {question_data['name']}, provide more detailed information and evidence.",
        "token_usage": {
            "evaluation": {"input": 0, "output": 0, "total": 0},
            "improvement": {"input": 0, "output": 0, "total": 0},
            "question_total": 0
        }
    }

