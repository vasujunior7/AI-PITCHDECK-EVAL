"""
💡 Improvement Router
Get AI-powered improvement suggestions
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from utils.file_handler import load_analysis_result, save_improvement_report, load_improvement_report
from services.recommendation_service import generate_recommendations
from core.logging import logger

router = APIRouter()

# Cache for improvement generation status
improvement_cache = {}


@router.post("/improve/{analysis_id}")
async def generate_improvement_suggestions(analysis_id: str, background_tasks: BackgroundTasks):
    """
    Generate AI-powered improvement suggestions
    
    This may take a few minutes as GPT-4 analyzes each slide
    """
    
    # Check if analysis exists
    try:
        result = load_analysis_result(analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found. Please analyze first.")
    
    # Check if already generated
    try:
        report = load_improvement_report(analysis_id)
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "message": "Improvements already generated",
            "slide_count": len(report.slide_improvements)
        }
    except FileNotFoundError:
        pass
    
    # Start generation in background
    improvement_cache[analysis_id] = {"status": "processing", "progress": 0}
    background_tasks.add_task(run_improvement_generation, analysis_id, result)
    
    return {
        "analysis_id": analysis_id,
        "status": "processing",
        "message": "Generating improvements. This may take a few minutes."
    }


@router.get("/improve/{analysis_id}/status")
async def get_improvement_status(analysis_id: str):
    """Check improvement generation status"""
    
    if analysis_id not in improvement_cache:
        # Check if report exists
        try:
            load_improvement_report(analysis_id)
            return {
                "analysis_id": analysis_id,
                "status": "completed",
                "progress": 100
            }
        except:
            raise HTTPException(status_code=404, detail="Improvement generation not started")
    
    return {
        "analysis_id": analysis_id,
        **improvement_cache[analysis_id]
    }


@router.get("/improve/{analysis_id}/report")
async def get_improvement_report(analysis_id: str):
    """Get complete improvement report"""
    
    try:
        report = load_improvement_report(analysis_id)
        return report
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Improvement report not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load report: {str(e)}")


@router.get("/improve/{analysis_id}/slide/{slide_number}")
async def get_slide_improvement(analysis_id: str, slide_number: int):
    """Get improvement suggestion for specific slide"""
    
    try:
        report = load_improvement_report(analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Improvement report not found")
    
    # Find slide improvement
    improvement = next((imp for imp in report.slide_improvements 
                       if imp.slide_number == slide_number), None)
    
    if not improvement:
        raise HTTPException(status_code=404, detail=f"No improvement found for slide {slide_number}")
    
    return improvement


async def run_improvement_generation(analysis_id: str, result):
    """Background task: Generate improvements"""
    
    logger.info(f"💡 [{analysis_id}] Generating improvements...")
    
    try:
        # Ensure slides is a list and scoring_result is a dict
        slides = result.get("slides", [])
        # Remove slides from result to get scoring_result
        scoring_result = {k: v for k, v in result.items() if k != "slides"}
        report = generate_recommendations(slides, scoring_result, max_recommendations=10)
        save_improvement_report(analysis_id, report)
        improvement_cache[analysis_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Improvements generated successfully"
        }
        logger.info(f"✅ [{analysis_id}] Improvements generated for {len(report.get('slide_improvements', []))} slides")
    except Exception as e:
        logger.error(f"❌ [{analysis_id}] Failed to generate improvements: {str(e)}")
        improvement_cache[analysis_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"Failed: {str(e)}"
        }


def update_progress(analysis_id: str, progress: int):
    """Update improvement generation progress"""
    if analysis_id in improvement_cache:
        improvement_cache[analysis_id]["progress"] = progress
