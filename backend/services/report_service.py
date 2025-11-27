"""
📄 Report Service - Unified Report Generation
Supports both PDF and PowerPoint report formats
"""

from .pdf_report_service import generate_pdf_report_from_analysis as generate_pdf
from .ppt_report_service import generate_ppt_report as generate_ppt
from pathlib import Path


def generate_report(analysis_data: dict, format: str = "pdf", output_path: str = None) -> str:
    """
    Generate comprehensive analysis report
    
    Args:
        analysis_data: Complete analysis results dictionary
        format: Report format - "pdf" or "ppt" (default: "pdf")
        output_path: Optional custom output path
        
    Returns:
        Path to generated report file
        
    Raises:
        ValueError: If format is not supported
    """
    format = format.lower()
    
    if format == "pdf":
        return generate_pdf(analysis_data, output_path)
    elif format in ["ppt", "pptx", "powerpoint"]:
        return generate_ppt(analysis_data, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'pdf' or 'ppt'.")


def generate_both_reports(analysis_data: dict, output_dir: str = None) -> dict:
    """
    Generate both PDF and PowerPoint reports
    
    Args:
        analysis_data: Complete analysis results dictionary
        output_dir: Optional output directory for both reports
        
    Returns:
        Dictionary with paths to both reports
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = str(output_dir / f"analysis_report_{timestamp}.pdf")
        ppt_path = str(output_dir / f"analysis_report_{timestamp}.pptx")
    else:
        pdf_path = None
        ppt_path = None
    
    return {
        "pdf": generate_pdf(analysis_data, pdf_path),
        "ppt": generate_ppt(analysis_data, ppt_path)
    }


# Legacy function aliases for backward compatibility
def generate_pdf_report(analysis_data: dict, output_path: str = None) -> str:
    """Generate PDF report (legacy alias)"""
    return generate_pdf(analysis_data, output_path)


def generate_ppt_report(analysis_data: dict, output_path: str = None) -> str:
    """Generate PowerPoint report (legacy alias)"""
    return generate_ppt(analysis_data, output_path)
