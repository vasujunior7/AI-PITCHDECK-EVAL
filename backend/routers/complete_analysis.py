"""
🚀 Complete Analysis Router - Full Pipeline
Upload PPT → Parse → Analyze → Score → Generate Reports
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import asyncio
from pathlib import Path
import shutil
from datetime import datetime

from services.parsing_service import parse_slides
from services.preprocessing_service import preprocess_slides
from services.classification_service import classify_slides
from services.feature_extraction_service import extract_features
from services.ai_evaluation_service import evaluate_slides as ai_evaluate_slides
from services.milestone3_evaluation_service_smart import evaluate_milestone3  # Smart fallback: OpenAI → Groq
from services.scoring_service import calculate_presentation_score
from services.recommendation_service import generate_recommendations
from services.report_service import generate_both_reports
from core.logging import logger

router = APIRouter(prefix="/api/analyze", tags=["analysis"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/full")
async def analyze_presentation_full(file: UploadFile = File(...)):
    """
    Complete analysis pipeline:
    1. Upload & Parse presentation
    2. Preprocess text
    3. Classify sections
    4. Extract features
    5. AI evaluation
    6. Calculate scores
    7. Generate recommendations
    8. Create reports (PDF + PPT)
    
    Returns analysis results with download links
    """
    
    logger.info(f"Starting full analysis for: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith(('.pptx', '.pdf')):
        raise HTTPException(status_code=400, detail="Only PPTX and PDF files are supported")
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved: {file_path}")
        
        # Step 1: Parse
        logger.info("Step 1/7: Parsing slides...")
        slides = parse_slides(str(file_path))
        logger.info(f"Parsed {len(slides)} slides")
        
        # Step 2: Preprocess
        logger.info("Step 2/7: Preprocessing...")
        preprocessed_slides = preprocess_slides(slides)
        
        # Step 3: Classify
        logger.info("Step 3/7: Classifying sections...")
        classified_slides = classify_slides(preprocessed_slides)
        avg_confidence = sum(s.section_confidence for s in classified_slides) / len(classified_slides) * 100
        logger.info(f"Classification confidence: {avg_confidence:.1f}%")
        
        # Step 4: Extract Features
        logger.info("Step 4/7: Extracting features...")
        features = extract_features(classified_slides)
        
        # Step 5: AI Evaluation
        logger.info("Step 5/8: AI evaluation...")
        ai_evaluation = await ai_evaluate_slides(classified_slides)
        
        # Step 5.5: Milestone 3 Evaluation (Venture Assessment - 90% weight)
        # Smart fallback: tries OpenAI first, falls back to Groq if quota/rate limit errors
        logger.info("Step 5.5/8: Milestone 3 Venture Assessment evaluation...")
        milestone3_evaluation = await evaluate_milestone3(classified_slides)
        milestone3_score = milestone3_evaluation.get('overall_score', 0)
        logger.info(f"Milestone 3 overall score: {milestone3_score:.1f}/100")
        
        # Step 6: Calculate Scores (combines Milestone 3 90% + Old PPT 10%)
        logger.info("Step 6/8: Calculating combined scores...")
        scoring_result = await calculate_presentation_score(classified_slides, milestone3_score=milestone3_score)
        
        # Step 7: Generate Recommendations
        logger.info("Step 7/7: Generating recommendations...")
        recommendations = generate_recommendations(
            slides=classified_slides,
            scoring_result=scoring_result,
            features=features
        )
        
        # Prepare analysis data
        analysis_data = {
            "presentation_name": Path(file.filename).stem,
            "overall_score": scoring_result['overall_score'],
            "grade": scoring_result['grade'],
            "component_scores": [
                {"component": comp, "score": score}
                for comp, score in scoring_result.get('component_scores', {}).items()
            ],
            "per_slide_scores": scoring_result.get('per_slide_scores', []),
            "feature_analysis": features,
            "ai_evaluation": ai_evaluation,
            "milestone3_evaluation": milestone3_evaluation,
            "recommendations": recommendations
        }
        
        # Generate reports
        logger.info("Generating reports...")
        report_paths = generate_both_reports(analysis_data)
        
        logger.info("Analysis complete!")
        
        # Return comprehensive results
        return {
            "success": True,
            "presentation_name": Path(file.filename).stem,
            "total_slides": len(classified_slides),
            "overall_score": scoring_result['overall_score'],
            "grade": scoring_result['grade'],
            "classification_confidence": round(avg_confidence, 2),
            "component_scores": scoring_result.get('component_scores', {}),
            "per_slide_scores": scoring_result.get('per_slide_scores', []),
            "feature_analysis": features,
            "ai_evaluation": {
                "overall_score": ai_evaluation.get('overall_score', 0),
                "strengths": ai_evaluation.get('strengths', [])[:5],
                "areas_for_improvement": ai_evaluation.get('areas_for_improvement', [])[:5]
            },
            "recommendations": [
                {
                    "priority": rec.get('priority', 'medium'),
                    "description": rec.get('description', rec.get('message', '')),
                    "impact": rec.get('impact', 'N/A')
                }
                for rec in recommendations['recommendations'][:20]
            ],
            "reports": {
                "pdf": str(report_paths.get('pdf', '')),
                "ppt": str(report_paths.get('ppt', ''))
            },
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Cleanup uploaded file (optional)
        # file_path.unlink(missing_ok=True)
        pass


@router.get("/download/pdf/{filename}")
async def download_pdf_report(filename: str):
    """Download generated PDF report"""
    file_path = Path("outputs/reports") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename
    )


@router.get("/download/ppt/{filename}")
async def download_ppt_report(filename: str):
    """Download generated PowerPoint report"""
    file_path = Path("outputs/reports") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=file_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=filename
    )


@router.get("/status")
async def get_status():
    """Check API status"""
    return {
        "status": "online",
        "service": "Presentation Analysis API",
        "version": "1.0.0",
        "features": [
            "Upload PPT/PDF",
            "Parse slides",
            "Classify sections",
            "Extract features",
            "AI evaluation",
            "Scoring with academic grading",
            "Recommendations",
            "PDF & PowerPoint reports"
        ]
    }
