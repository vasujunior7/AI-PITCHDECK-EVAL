"""
📁 File Handler
Save and load analysis results
"""

import json
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
from models.analysis import AnalysisResult
from models.report import ImprovementReport
from core.config import settings


def save_analysis_result(analysis_id: str, result: AnalysisResult):
    """Save analysis result to JSON file"""
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    file_path = results_dir / f"{analysis_id}.json"
    
    # Convert to dict and save
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result.model_dump(), f, indent=2, default=str)


def load_analysis_result(analysis_id: str) -> AnalysisResult:
    """Load analysis result from JSON file"""
    
    file_path = Path("results") / f"{analysis_id}.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Analysis result not found: {analysis_id}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return AnalysisResult(**data)


def save_improvement_report(analysis_id: str, report: ImprovementReport):
    """Save improvement report to JSON file"""
    
    improvements_dir = Path("improvements")
    improvements_dir.mkdir(exist_ok=True)
    
    file_path = improvements_dir / f"{analysis_id}.json"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(report.model_dump(), f, indent=2, default=str)


def load_improvement_report(analysis_id: str) -> ImprovementReport:
    """Load improvement report from JSON file"""
    
    file_path = Path("improvements") / f"{analysis_id}.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Improvement report not found: {analysis_id}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return ImprovementReport(**data)


def cleanup_old_files(days: int = 7):
    """Delete files older than specified days"""
    
    cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
    
    for directory in ["uploads", "results", "improvements", settings.REPORTS_DIR]:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        
        for file in dir_path.iterdir():
            if file.is_file() and file.stat().st_mtime < cutoff:
                file.unlink()
                print(f"Deleted old file: {file}")
