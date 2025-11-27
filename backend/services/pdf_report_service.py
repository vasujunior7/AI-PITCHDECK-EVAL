"""
Milestone 9: PDF Report Generation
Generates professional PDF reports with analysis results, charts, and recommendations
"""

from datetime import datetime
from typing import Dict, List, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.spider import SpiderChart
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

matplotlib.use('Agg')  # Non-interactive backend

from core.logging import logger


def _get_rating(score: float) -> str:
    if score >= 75: return "Excellent"
    elif score >= 55: return "Very Good"
    elif score >= 40: return "Good"
    elif score >= 25: return "Fair"
    else: return "Needs Improvement"


def _get_density_interpretation(density: float) -> str:
    if density >= 0.8: return "Excellent diversity of concepts"
    elif density >= 0.6: return "Good conceptual variety"
    elif density >= 0.4: return "Moderate concept repetition"
    else: return "High concept repetition"


def _get_bloom_interpretation(level: float) -> str:
    bloom_levels = {
        1: "Remember (basic recall)",
        2: "Understand (comprehension)",
        3: "Apply (practical application)",
        4: "Analyze (critical thinking)",
        5: "Evaluate (judgment)",
        6: "Create (synthesis)"
    }
    return bloom_levels.get(int(round(level)), "Unknown")


def _get_redundancy_interpretation(redundancy: float) -> str:
    if redundancy < 0.1: return "Excellent - minimal repetition"
    elif redundancy < 0.3: return "Good - acceptable repetition"
    elif redundancy < 0.5: return "Fair - some redundancy present"
    else: return "Poor - significant redundancy"


def generate_pdf_report(
    slides: List,
    scoring_result: Dict,
    recommendations: Dict,
    milestone3_evaluation: Dict = None,
    filename: str = "presentation_analysis_report.pdf"
) -> bytes:
    """
    Generate comprehensive PDF report
    """
    logger.info(f"Generating PDF report: {filename}")
    
    # Create buffer for PDF
    buffer = BytesIO()
    
    # Create document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    # Container for elements
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=12,
        spaceBefore=20,
        borderPadding=5,
        borderColor=colors.HexColor('#BDC3C7'),
        borderWidth=0,
        borderBottomWidth=1
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=10,
        spaceBefore=10
    )
    
    # 1. Title Page
    story.extend(_create_title_page(scoring_result, styles, title_style))
    story.append(PageBreak())
    
    # 2. Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.extend(_create_executive_summary(scoring_result, styles))
    story.append(Spacer(1, 24))
    
    # 3. Venture Builder Assessment (New Section)
    if milestone3_evaluation:
        story.append(Paragraph("Venture Builder Assessment", heading_style))
        story.append(Paragraph("Evaluation of 18 key venture questions across 6 categories.", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Overall Venture Score
        story.extend(_create_venture_score_section(milestone3_evaluation, styles))
        story.append(Spacer(1, 12))
        
        # 3 Spider Charts
        story.append(Paragraph("Category Performance Analysis", subheading_style))
        story.extend(_create_spider_charts(milestone3_evaluation))
        story.append(Spacer(1, 12))
        
        # Category Bar Chart (New)
        story.append(Paragraph("Venture Category Scores", subheading_style))
        story.extend(_create_venture_bar_chart(milestone3_evaluation))
        story.append(Spacer(1, 12))
        
        # Detailed Questions Table
        story.append(Paragraph("Detailed Question Analysis", subheading_style))
        story.extend(_create_venture_questions_table(milestone3_evaluation, styles))
        story.append(Spacer(1, 12))
        
        # Venture Recommendations (if any)
        # Assuming recommendations might contain venture specific ones, or we generate generic ones based on score
        story.append(PageBreak())

    # 4. Presentation Analysis (Old PPT Results)
    story.append(Paragraph("Presentation Quality Analysis", heading_style))
    story.append(Paragraph("Analysis of slide structure, design, clarity, and content quality.", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Overall PPT Score
    story.append(Paragraph("Overall Presentation Score", subheading_style))
    story.extend(_create_score_section(scoring_result, styles))
    story.append(Spacer(1, 12))
    
    # Component Scores
    story.append(Paragraph("Component Breakdown", subheading_style))
    story.extend(_create_component_analysis(scoring_result, styles))
    story.append(Spacer(1, 12))
    
    # Feature Analysis
    story.append(Paragraph("Feature Analysis", subheading_style))
    story.extend(_create_feature_analysis(scoring_result, styles, subheading_style))
    story.append(PageBreak())
    
    # Slide-by-Slide Breakdown
    story.append(Paragraph("Slide-by-Slide Analysis", heading_style))
    story.extend(_create_slide_breakdown(scoring_result, styles))
    story.append(PageBreak())
    
    # 5. Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    story.extend(_create_recommendations_section(recommendations, styles, subheading_style))
    story.append(PageBreak())
    
    # 6. Strengths & Weaknesses
    story.append(Paragraph("Strengths & Areas for Improvement", heading_style))
    story.extend(_create_strengths_weaknesses(scoring_result, styles, subheading_style))
    
    # Build PDF
    doc.build(story)
    
    # Get PDF content
    pdf_content = buffer.getvalue()
    buffer.close()
    
    logger.info(f"PDF report generated successfully ({len(pdf_content)} bytes)")
    return pdf_content


def _create_title_page(scoring_result: Dict, styles, title_style) -> List:
    """Create title page"""
    elements = []
    
    elements.append(Spacer(1, 2*inch))
    elements.append(Paragraph("Venture Pitch Analysis Report", title_style))
    elements.append(Spacer(1, 0.5*inch))
    
    score = scoring_result.get("overall_score", 0)
    grade = scoring_result.get("grade", "N/A")
    
    score_text = f"""
    <para align=center>
        <font size=48 color="#2ECC71">{score:.1f}</font>
        <font size=24 color="#7F8C8D">/100</font><br/>
        <font size=32 color="#3498DB">Grade: {grade}</font>
    </para>
    """
    elements.append(Paragraph(score_text, styles['Normal']))
    elements.append(Spacer(1, 1*inch))
    
    date_text = f"<para align=center><font size=12 color='#7F8C8D'>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</font></para>"
    elements.append(Paragraph(date_text, styles['Normal']))
    
    return elements


def _create_executive_summary(scoring_result: Dict, styles) -> List:
    """Create executive summary section"""
    elements = []
    
    score = scoring_result.get("overall_score", 0)
    grade = scoring_result.get("grade", "N/A")
    stats = scoring_result.get("statistics", {})
    
    summary_text = f"""
    This report provides a comprehensive analysis of your pitch deck, evaluating both the <b>venture potential</b> 
    (business model, market, execution) and the <b>presentation quality</b> (design, clarity, structure).
    <br/><br/>
    The overall score of <b>{score:.1f}/100 (Grade {grade})</b> is a weighted combination of these factors.
    """
    
    elements.append(Paragraph(summary_text, styles['Normal']))
    
    return elements


def _create_venture_score_section(m3_eval: Dict, styles) -> List:
    """Create venture score section - DISABLED to show only overall score"""
    # User requested to show only overall score, not separate venture score
    return []


def _create_spider_charts(m3_eval: Dict) -> List:
    """Create 3 spider charts for venture categories"""
    elements = []
    
    # Extract category scores
    cat_scores = m3_eval.get("category_scores", {})
    
    # Define chart groups
    charts_config = [
        {
            "title": "Problem & Solution",
            "categories": ["Problem & Solution", "Market Analysis"], # Grouping for chart 1
            "labels": ["Problem", "Solution", "Market Size", "Competition", "Timing", "Why Now"] # Mock labels if detailed not avail
        },
        {
            "title": "Business & Product",
            "categories": ["Business Model", "Product"],
            "labels": ["Revenue", "Pricing", "Product", "Tech", "Roadmap", "IP"]
        },
        {
            "title": "Execution & Ask",
            "categories": ["Go-To-Market", "Execution", "Presentation"],
            "labels": ["GTM", "Sales", "Team", "Financials", "The Ask", "Exit"]
        }
    ]
    
    # Since we only have high level category scores in m3_eval['category_scores'], 
    # we might need to use individual question scores if available or just plot the categories.
    # The user asked for "3 spider charts 6 pairs of questions each".
    # Let's try to map the 18 questions to 3 charts of 6 questions each.
    
    evaluations = m3_eval.get("evaluations", [])
    if not evaluations or len(evaluations) < 18:
        # Fallback if we don't have all questions
        return [Paragraph("Insufficient data for detailed spider charts.", getSampleStyleSheet()['Normal'])]

    # Sort evaluations by question ID just in case
    # Assuming evaluations list order matches questions 1-18 or has 'id'
    # Let's try to map by index if 'id' is missing or just take first 18
    
    # Chart 1: Questions 1-6
    # Chart 2: Questions 7-12
    # Chart 3: Questions 13-18
    
    chart_groups = [
        ("Market & Problem (Q1-Q6)", evaluations[0:6]),
        ("Business & Product (Q7-Q12)", evaluations[6:12]),
        ("Execution & Financials (Q13-Q18)", evaluations[12:18])
    ]
    
    for title, group_evals in chart_groups:
        if not group_evals: continue
        
        labels = [f"Q{e.get('id', i+1)}" for i, e in enumerate(group_evals)]
        data = [e.get('score', 0) for e in group_evals]
        
        # Create radar chart using matplotlib
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        N = len(labels)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1] # Close the loop
        
        # Data to plot
        values = data + data[:1] # Close the loop
        
        # Plot
        ax.plot(angles, values, linewidth=1, linestyle='solid', color='#3498DB')
        ax.fill(angles, values, '#3498DB', alpha=0.2)
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        
        # Y-labels
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=7)
        plt.ylim(0, 100)
        
        plt.title(title, size=10, y=1.1)
        
        # Save
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        
        img = Image(img_buffer, width=3*inch, height=3*inch)
        elements.append(img)
        
    return elements


def _create_venture_questions_table(m3_eval: Dict, styles) -> List:
    """Create detailed table for venture questions"""
    elements = []
    
    evaluations = m3_eval.get("evaluations", [])
    
    data = [['ID', 'Question', 'Score', 'Analysis & Improvement']]
    
    col_widths = [0.4*inch, 2.0*inch, 0.6*inch, 4.0*inch]
    
    for eval_item in evaluations:
        q_id = str(eval_item.get('id', ''))
        question = eval_item.get('question', '')
        score = eval_item.get('score', 0)
        
        justification = eval_item.get('justification', eval_item.get('reasoning', ''))
        suggestion = eval_item.get('improvement_suggestion', '')
        
        # Combine justification and suggestion
        combined_text = f"<b>Analysis:</b> {justification}<br/><br/><b>Improvement:</b> {suggestion}"
        
        data.append([
            q_id,
            Paragraph(question, styles['Normal']),
            f"{score:.0f}",
            Paragraph(combined_text, styles['Normal'])
        ])
    
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'), # ID center
        ('ALIGN', (2, 0), (2, -1), 'CENTER'), # Score center
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9F9')]),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    
    elements.append(table)
    return elements


def _create_score_section(scoring_result: Dict, styles) -> List:
    """Create overall score section with gauge chart"""
    elements = []
    
    score = scoring_result.get("overall_score", 0)
    
    # Create matplotlib gauge chart
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Create semi-circle gauge
    categories = ['F\n0-59', 'D\n60-69', 'C\n70-79', 'B\n80-89', 'A\n90-100']
    values = [10, 10, 10, 10, 10]  # Equal segments
    colors_list = ['#E74C3C', '#E67E22', '#F39C12', '#3498DB', '#2ECC71']
    
    wedges, texts = ax.pie(
        values,
        colors=colors_list,
        startangle=180,
        counterclock=False,
        wedgeprops={'width': 0.3}
    )
    
    # Add score indicator
    ax.text(0, 0, f'{score:.1f}', ha='center', va='center', fontsize=32, fontweight='bold')
    
    # Save to buffer
    img_buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    img_buffer.seek(0)
    
    # Add to PDF
    img = Image(img_buffer, width=4*inch, height=2*inch)
    elements.append(img)
    
    return elements


def _create_component_analysis(scoring_result: Dict, styles) -> List:
    """Create component scores table and chart"""
    elements = []
    
    components = scoring_result.get("component_scores", {})
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    
    comp_names = list(components.keys())
    comp_scores = list(components.values())
    
    # Color bars by score
    bar_colors = ['#2ECC71' if s >= 80 else '#F39C12' if s >= 60 else '#E74C3C' for s in comp_scores]
    
    bars = ax.barh(comp_names, comp_scores, color=bar_colors, alpha=0.8)
    ax.set_xlabel('Score', fontsize=10)
    ax.set_xlim(0, 100)
    ax.axvline(x=60, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=80, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, comp_scores)):
        ax.text(score + 2, i, f'{score:.1f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save to buffer
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    img_buffer.seek(0)
    
    # Add to PDF
    img = Image(img_buffer, width=5*inch, height=2.5*inch)
    elements.append(img)
    elements.append(Spacer(1, 12))
    
    # Table with detailed scores
    data = [['Component', 'Score', 'Rating']]
    for comp, score in components.items():
        rating = _get_rating(score)
        data.append([comp.replace('_', ' ').title(), f'{score:.1f}', rating])
    
    table = Table(data, colWidths=[2.5*inch, 1*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(table)
    
    return elements


def _create_feature_analysis(scoring_result: Dict, styles, subheading_style) -> List:
    """Create feature analysis section"""
    elements = []
    
    features = scoring_result.get("feature_analysis", {})
    
    if not isinstance(features, dict) or not features:
        features = {}
    
    def safe_extract_float(value, default=0):
        if isinstance(value, dict):
            for key in ["overall_density", "overall_quality", "average_level", "redundancy_score", "score"]:
                if key in value and not isinstance(value[key], dict):
                    try:
                        return float(value[key])
                    except (ValueError, TypeError):
                        continue
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # Semantic Density
    elements.append(Paragraph("Semantic Density", subheading_style))
    density = safe_extract_float(features.get("semantic_density", 0))
    density_text = f"<b>{density:.3f}</b> - {_get_density_interpretation(density)}"
    elements.append(Paragraph(density_text, styles['Normal']))
    elements.append(Spacer(1, 8))
    
    # Layout Quality
    elements.append(Paragraph("Layout Quality", subheading_style))
    layout = safe_extract_float(features.get("layout_quality", 0))
    layout_text = f"<b>{layout:.1f}/100</b> - {_get_rating(layout)}"
    elements.append(Paragraph(layout_text, styles['Normal']))
    elements.append(Spacer(1, 8))
    
    # Bloom's Taxonomy
    elements.append(Paragraph("Cognitive Engagement (Bloom's Taxonomy)", subheading_style))
    bloom = safe_extract_float(features.get("avg_bloom_level", 0))
    bloom_text = f"<b>Level {bloom:.2f}/6</b> - {_get_bloom_interpretation(bloom)}"
    elements.append(Paragraph(bloom_text, styles['Normal']))
    elements.append(Spacer(1, 8))
    
    # Redundancy
    elements.append(Paragraph("Content Redundancy", subheading_style))
    redundancy = safe_extract_float(features.get("redundancy_score", 0))
    redundancy_text = f"<b>{redundancy:.1%}</b> - {_get_redundancy_interpretation(redundancy)}"
    elements.append(Paragraph(redundancy_text, styles['Normal']))
    
    return elements


def _create_slide_breakdown(scoring_result: Dict, styles) -> List:
    """Create slide-by-slide breakdown table"""
    elements = []
    
    slide_scores = scoring_result.get("slide_scores", [])
    
    data = [['Slide', 'Score', 'Clarity', 'Structure', 'Depth', 'Design', 'Rating']]
    
    for slide in slide_scores:
        data.append([
            str(slide['slide_number']),
            f"{slide['score']:.1f}",
            f"{slide['clarity']:.0f}",
            f"{slide['structure']:.0f}",
            f"{slide['depth']:.0f}",
            f"{slide['design']:.0f}",
            _get_rating(slide['score'])
        ])
    
    table = Table(data, colWidths=[0.6*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.7*inch, 0.8*inch, 1.2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(table)
    
    return elements


def _create_recommendations_section(recommendations: Dict, styles, subheading_style) -> List:
    """Create recommendations section"""
    elements = []
    
    recs = recommendations.get("recommendations", [])
    if not isinstance(recs, list):
        recs = []
        
    impact = recommendations.get("estimated_impact", {})
    
    impact_text = f"""
    <b>Potential Impact:</b> Implementing these recommendations could improve your score by up to 
    <font color="#2ECC71"><b>+{impact.get('total_points', 0)} points</b></font>
    """
    elements.append(Paragraph(impact_text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    priority_colors = {
        'critical': '#E74C3C',
        'high': '#E67E22',
        'medium': '#F39C12',
        'low': '#2ECC71'
    }
    
    for i, rec in enumerate(recs[:20], 1):
        priority = rec.get('priority', 'medium')
        color = priority_colors.get(priority, '#7F8C8D')
        
        rec_text = f"""
        <para>
        <font size=11 color="{color}"><b>[{priority.upper()}]</b></font> 
        <b>{rec.get('category', 'General')}:</b> {rec.get('recommendation', '')}<br/>
        <font size=9 color="#7F8C8D">Impact: +{rec.get('impact', 0)} points | 
        Effort: {rec.get('effort', 'medium').title()}</font>
        </para>
        """
        elements.append(Paragraph(rec_text, styles['Normal']))
        elements.append(Spacer(1, 8))
    
    return elements


def _create_strengths_weaknesses(scoring_result: Dict, styles, subheading_style) -> List:
    """Create strengths and weaknesses section"""
    elements = []
    
    strengths = scoring_result.get("strengths", [])
    weaknesses = scoring_result.get("weaknesses", [])
    
    elements.append(Paragraph("💪 Strengths", subheading_style))
    if strengths:
        for strength in strengths:
            elements.append(Paragraph(f"• {strength}", styles['Normal']))
    else:
        elements.append(Paragraph("No major strengths identified.", styles['Normal']))
    
    elements.append(Spacer(1, 16))
    
    elements.append(Paragraph("⚠️ Areas for Improvement", subheading_style))
    if weaknesses:
        for weakness in weaknesses:
            elements.append(Paragraph(f"• {weakness}", styles['Normal']))
    else:
        elements.append(Paragraph("No significant weaknesses found.", styles['Normal']))
    
    return elements





def generate_pdf_report_from_analysis(analysis_data: dict, output_path: str = None) -> str:
    """
    Wrapper function to generate PDF from unified analysis data structure
    """
    from pathlib import Path
    
    # Extract data from unified format
    slides = []
    per_slide = analysis_data.get('per_slide_scores', [])
    for slide_data in per_slide:
        slides.append({
            'slide_number': slide_data.get('slide', 0),
            'text': '',
            'word_count': 0
        })
    
    # Build scoring result
    scoring_result = {
        'overall_score': analysis_data.get('overall_score', 0),
        'grade': analysis_data.get('grade', 'N/A'),
        'component_scores': {
            comp.get('component', 'Unknown'): comp.get('score', 0)
            for comp in analysis_data.get('component_scores', [])
        },
        'per_slide_scores': per_slide,
        'slide_scores': per_slide,
        'feature_analysis': analysis_data.get('feature_analysis', {}),
        'ai_evaluation': analysis_data.get('ai_evaluation', {}),
        'strengths': analysis_data.get('strengths', []),
        'weaknesses': analysis_data.get('weaknesses', []),
        'statistics': analysis_data.get('statistics', {})
    }
    
    # Extract Milestone 3 evaluation
    milestone3_evaluation = analysis_data.get('milestone3_evaluation', {})
    
    # Build recommendations
    recommendations_dict = analysis_data.get('recommendations', {})
    if not isinstance(recommendations_dict, dict):
        recommendations_dict = {'recommendations': []}
    
    # Determine output path
    if output_path is None:
        output_dir = Path("outputs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"analysis_report_{timestamp}.pdf"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate PDF
    pdf_bytes = generate_pdf_report(
        slides, 
        scoring_result, 
        recommendations_dict, 
        milestone3_evaluation,
        str(output_path.name)
    )
    
    # Write to file
    with open(output_path, 'wb') as f:
        f.write(pdf_bytes)
    
    return str(output_path)


def _create_venture_bar_chart(m3_eval: Dict) -> List:
    """Create bar chart for venture categories"""
    elements = []
    
    cat_scores = m3_eval.get("category_scores", {})
    if not cat_scores:
        return []
        
    # Create bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    
    categories = list(cat_scores.keys())
    scores = list(cat_scores.values())
    
    # Color bars by score
    bar_colors = ['#2ECC71' if s >= 80 else '#F39C12' if s >= 60 else '#E74C3C' for s in scores]
    
    bars = ax.barh(categories, scores, color=bar_colors, alpha=0.8)
    ax.set_xlabel('Score', fontsize=10)
    ax.set_xlim(0, 100)
    ax.axvline(x=60, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=80, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 2, i, f'{score:.1f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save to buffer
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    img_buffer.seek(0)
    
    # Add to PDF
    img = Image(img_buffer, width=5*inch, height=2.5*inch)
    elements.append(img)
    
    return elements



