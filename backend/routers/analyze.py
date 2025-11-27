"""
🔍 Analysis Router
Main analysis orchestration endpoint
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
from datetime import datetime
import json

from core.config import settings
from core.logging import logger
from models.analysis import AnalysisStatus, AnalysisResult
from services.parsing_service import parse_slides
from services.preprocessing_service import preprocess_slides
from services.classification_service import classify_slides
from services.feature_service import extract_features
from services.evaluation_service import evaluate_slides
# from services.scoring_service import aggregate_scores
from utils.file_handler import save_analysis_result, load_analysis_result

router = APIRouter()

# In-memory storage for analysis status (in production, use a database)
analysis_cache = {}


@router.post("/analyze/{analysis_id}")
async def analyze_presentation(analysis_id: str, background_tasks: BackgroundTasks):
    """
    Start comprehensive presentation analysis
    
    Pipeline:
    1. Parse slides
    2. Preprocess text
    3. Classify sections
    4. Extract advanced features
    5. AI evaluation
    6. Score aggregation
    """
    
    # Check if file exists
    file_path = None
    for ext in settings.ALLOWED_EXTENSIONS:
        potential_path = Path(settings.UPLOAD_DIR) / f"{analysis_id}{ext}"
        if potential_path.exists():
            file_path = potential_path
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found. Please upload first.")
    
    # Check if already analyzed
    if analysis_id in analysis_cache and analysis_cache[analysis_id].status == "completed":
        logger.info(f"♻️ Analysis {analysis_id} already completed")
        return load_analysis_result(analysis_id)
    
    # Initialize status
    analysis_cache[analysis_id] = AnalysisStatus(
        analysis_id=analysis_id,
        status="processing",
        progress=0,
        message="Starting analysis..."
    )
    
    # Start analysis in background
    background_tasks.add_task(run_analysis, analysis_id, file_path)
    
    return {
        "analysis_id": analysis_id,
        "status": "processing",
        "message": "Analysis started. Use /analyze/{analysis_id}/status to check progress."
    }


@router.get("/analyze/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    """Get current analysis status"""
    
    if analysis_id not in analysis_cache:
        # Check if result exists on disk
        try:
            result = load_analysis_result(analysis_id)
            return {
                "analysis_id": analysis_id,
                "status": "completed",
                "progress": 100,
                "message": "Analysis completed"
            }
        except:
            raise HTTPException(status_code=404, detail="Analysis not found")
    
    status = analysis_cache[analysis_id]
    return status


@router.get("/analyze/{analysis_id}/result")
async def get_analysis_result(analysis_id: str):
    """Get complete analysis result"""
    
    try:
        result = load_analysis_result(analysis_id)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    except Exception as e:
        logger.error(f"Error loading analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load analysis result")


async def run_analysis(analysis_id: str, file_path: Path):
    """
    Background task: Run complete analysis pipeline
    """
    
    logger.info(f"🚀 Starting analysis for {analysis_id}")
    
    try:
        # Step 1: Parse slides (20%)
        analysis_cache[analysis_id].progress = 10
        analysis_cache[analysis_id].message = "Parsing slides..."
        logger.info(f"📄 [{analysis_id}] Parsing slides...")
        
        slides = parse_slides(str(file_path))
        analysis_cache[analysis_id].progress = 20
        logger.info(f"✅ [{analysis_id}] Parsed {len(slides)} slides")
        
        # Step 2: Preprocess text (40%)
        analysis_cache[analysis_id].progress = 25
        analysis_cache[analysis_id].message = "Preprocessing text..."
        logger.info(f"🧹 [{analysis_id}] Preprocessing text...")
        
        slides = preprocess_slides(slides)
        analysis_cache[analysis_id].progress = 40
        
        # Step 3: Classify sections (50%)
        analysis_cache[analysis_id].progress = 45
        analysis_cache[analysis_id].message = "Classifying sections..."
        logger.info(f"🏷️ [{analysis_id}] Classifying sections...")
        
        slides = classify_slides(slides)
        analysis_cache[analysis_id].progress = 50
        
        # Step 4: Extract features (70%)
        analysis_cache[analysis_id].progress = 55
        analysis_cache[analysis_id].message = "Extracting features..."
        logger.info(f"🧠 [{analysis_id}] Extracting advanced features...")
        
        features = extract_features(slides)
        analysis_cache[analysis_id].progress = 70
        
        # Step 5: AI evaluation (85%)
        analysis_cache[analysis_id].progress = 75
        analysis_cache[analysis_id].message = "AI evaluation..."
        logger.info(f"🤖 [{analysis_id}] Running AI evaluation...")
        
        slide_scores = evaluate_slides(slides, features)
        analysis_cache[analysis_id].progress = 85
        
        # Step 6: Aggregate scores (95%)
        analysis_cache[analysis_id].progress = 90
        analysis_cache[analysis_id].message = "Calculating scores..."
        logger.info(f"📊 [{analysis_id}] Aggregating scores...")
        
        result = aggregate_scores(analysis_id, slides, features, slide_scores)
        analysis_cache[analysis_id].progress = 95
        
        # Save result
        analysis_cache[analysis_id].message = "Saving results..."
        save_analysis_result(analysis_id, result)
        
        # Complete
        analysis_cache[analysis_id].status = "completed"
        analysis_cache[analysis_id].progress = 100
        analysis_cache[analysis_id].message = "Analysis completed successfully!"
        
        logger.info(f"🎉 [{analysis_id}] Analysis completed! Score: {result.overall_score.final_score:.1f}/100")
        
    except Exception as e:
        logger.error(f"❌ [{analysis_id}] Analysis failed: {str(e)}")
        analysis_cache[analysis_id].status = "failed"
        analysis_cache[analysis_id].message = f"Analysis failed: {str(e)}"
        raise
