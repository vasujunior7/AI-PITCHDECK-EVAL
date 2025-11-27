"""
🎯 AI Pitch Deck Evaluator - Streamlit Frontend
Beautiful, modern UI for analyzing presentations
"""

import streamlit as st
import requests
import time
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="AI Pitch Deck Evaluator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, beautiful design
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card-like containers */
    .stAlert {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #667eea;
        padding: 20px;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Grade badge */
    .grade-badge {
        font-size: 3em;
        font-weight: bold;
        padding: 20px;
        border-radius: 50%;
        display: inline-block;
        width: 100px;
        height: 100px;
        line-height: 60px;
    }
    
    .grade-A { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .grade-B { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .grade-C { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    .grade-D { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }
    .grade-F { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); color: white; }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 15px 40px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"
ANALYZE_ENDPOINT = f"{API_BASE_URL}/api/analyze/full"

# Session state initialization
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'report_paths' not in st.session_state:
    st.session_state.report_paths = None


def analyze_presentation(uploaded_file):
    """Send file to FastAPI backend for analysis"""
    try:
        # Prepare file for upload
        files = {
            'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        
        # Send request to backend
        response = requests.post(
            ANALYZE_ENDPOINT,
            files=files,
            timeout=900  # 15 minute timeout (analysis can take time with many slides)
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            error_detail = response.json().get('detail', 'Unknown error occurred')
            return None, f"Error {response.status_code}: {error_detail}"
            
    except requests.exceptions.Timeout:
        return None, "⏱️ Analysis is taking longer than expected (>15 minutes). This usually happens with very large presentations (>30 slides). The backend may still be processing - check the FastAPI logs. Try with a smaller file or wait a bit and check the backend."
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Please ensure the FastAPI server is running on http://localhost:8000"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def display_results(result):
    """Display analysis results in a beautiful format"""
    
    # Header with overall score
    st.markdown("## 📊 Analysis Results")
    
    # Overall Score Card
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        score = result.get('overall_score', 0)
        grade = result.get('grade', 'N/A')
        
        # Grade badge with color
        grade_class = f"grade-{grade}"
        st.markdown(f"""
        <div style="text-align: center; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            <h3 style="color: #667eea; margin-bottom: 20px;">Overall Score</h3>
            <div class="grade-badge {grade_class}">{grade}</div>
            <h1 style="color: #667eea; margin-top: 20px;">{score:.1f}/100</h1>
            <p style="color: #666; font-size: 1.2em;">Presentation Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Metrics
    st.markdown("### 📈 Key Metrics")
    
    metrics_cols = st.columns(4)
    
    with metrics_cols[0]:
        st.metric(
            label="📄 Total Slides",
            value=result.get('total_slides', 0)
        )
    
    with metrics_cols[1]:
        st.metric(
            label="🎯 Grade",
            value=grade
        )
    
    with metrics_cols[2]:
        confidence = result.get('classification_confidence', 0)
        st.metric(
            label="📊 Confidence",
            value=f"{confidence}%"
        )
    
    with metrics_cols[3]:
        total_recs = len(result.get('recommendations', []))
        st.metric(
            label="💡 Recommendations",
            value=total_recs
        )
    
    st.markdown("---")
    
    # Component Scores
    st.markdown("### 🔍 Component Breakdown")
    
    component_scores = result.get('component_scores', {})
    if component_scores:
        # Create horizontal layout with max 5 columns per row
        items = list(component_scores.items())
        num_components = len(items)
        
        # Calculate rows needed (5 items per row)
        num_rows = (num_components + 4) // 5  # Ceiling division
        
        for row_idx in range(num_rows):
            start_idx = row_idx * 5
            end_idx = min(start_idx + 5, num_components)
            row_items = items[start_idx:end_idx]
            
            cols = st.columns(len(row_items))
            for idx, (component, score) in enumerate(row_items):
                with cols[idx]:
                    # Determine color based on score
                    if score >= 75:
                        color = "#2ecc71"  # Green
                    elif score >= 55:
                        color = "#f39c12"  # Orange
                    else:
                        color = "#e74c3c"  # Red
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #667eea; margin-bottom: 10px;">{component.replace('_', ' ').title()}</h4>
                        <div style="font-size: 2em; font-weight: bold; color: {color};">{score:.0f}</div>
                        <div style="color: #999; font-size: 0.9em;">/ 100</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # AI Evaluation
    ai_eval = result.get('ai_evaluation', {})
    if ai_eval:
        st.markdown("### 🤖 AI Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            strengths = ai_eval.get('strengths', [])
            if strengths:
                st.markdown("**💪 Strengths**")
                for strength in strengths:
                    st.success(f"✓ {strength}")
        
        with col2:
            improvements = ai_eval.get('areas_for_improvement', [])
            if improvements:
                st.markdown("**⚠️ Areas for Improvement**")
                for improvement in improvements:
                    st.warning(f"• {improvement}")
    
    st.markdown("---")
    
    # Recommendations
    recommendations = result.get('recommendations', [])
    if recommendations:
        st.markdown("### 💡 Top Recommendations")
        
        for idx, rec in enumerate(recommendations[:10], 1):
            priority = rec.get('priority', 'medium')
            description = rec.get('description', rec.get('recommendation', 'No description'))
            impact = rec.get('impact', 'N/A')
            
            # Priority badge color
            priority_colors = {
                'critical': '#e74c3c',
                'high': '#e67e22',
                'medium': '#f39c12',
                'low': '#2ecc71'
            }
            color = priority_colors.get(priority, '#95a5a6')
            
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid {color};">
                <span style="background: {color}; color: white; padding: 5px 10px; border-radius: 5px; font-size: 0.8em; font-weight: bold;">
                    {priority.upper()}
                </span>
                <p style="color: #333; margin: 10px 0; font-size: 1.1em;">{description}</p>
                <small style="color: #999;">Impact: {impact}</small>
            </div>
            """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 3em; margin-bottom: 10px;">🎯 AI Pitch Deck Evaluator</h1>
        <p style="font-size: 1.2em; color: rgba(255,255,255,0.9);">
            Upload your presentation and get instant AI-powered feedback
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📚 About")
        st.info("""
        This tool analyzes your pitch deck using advanced AI to provide:
        
        ✅ **Venture Assessment** - 18 criteria evaluation
        
        ✅ **Presentation Quality** - Design, clarity, structure
        
        ✅ **AI Insights** - Strengths & improvements
        
        ✅ **Actionable Recommendations** - Specific suggestions
        """)
        
        st.markdown("## ⚙️ Settings")
        st.markdown("**API Endpoint:**")
        st.code(ANALYZE_ENDPOINT, language="")
        
        # Test connection
        if st.button("🔌 Test Connection"):
            try:
                response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Connected to backend!")
                else:
                    st.error("❌ Backend returned error")
            except:
                st.error("❌ Cannot connect to backend. Please start the FastAPI server.")
        
        # Download options in sidebar if analysis is complete
        if st.session_state.get('analysis_complete') and st.session_state.get('report_paths'):
            st.markdown("---")
            st.markdown("## 📥 Downloads")
            
            report_paths = st.session_state.report_paths
            
            if report_paths.get('pdf') and Path(report_paths['pdf']).exists():
                with open(report_paths['pdf'], 'rb') as f:
                    st.download_button(
                        label="📄 PDF Report",
                        data=f.read(),
                        file_name=Path(report_paths['pdf']).name,
                        mime="application/pdf",
                        key="sidebar_pdf_dl"
                    )
            
            if report_paths.get('ppt') and Path(report_paths['ppt']).exists():
                with open(report_paths['ppt'], 'rb') as f:
                    st.download_button(
                        label="📊 PPT Summary",
                        data=f.read(),
                        file_name=Path(report_paths['ppt']).name,
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        key="sidebar_ppt_dl"
                    )
    
    # Main content
    if not st.session_state.analysis_complete:
        # File Upload Section
        st.markdown("### 📤 Upload Your Presentation")
        
        uploaded_file = st.file_uploader(
            "Choose a PPT or PDF file",
            type=['ppt', 'pptx', 'pdf'],
            help="Upload your pitch deck in PPT, PPTX, or PDF format"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.json(file_details)
            
            with col2:
                if st.button("🚀 Analyze Presentation", use_container_width=True):
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Animate progress
                    for i in range(20):
                        progress_bar.progress(i * 5)
                        status_text.text(f"Uploading... {i * 5}%")
                        time.sleep(0.05)
                    
                    status_text.text("🔍 Analyzing presentation...")
                    
                    # Call API
                    result, error = analyze_presentation(uploaded_file)
                    
                    if error:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ {error}")
                    else:
                        # Complete progress
                        for i in range(20, 101):
                            progress_bar.progress(i)
                            time.sleep(0.01)
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Store results
                        st.session_state.analysis_complete = True
                        st.session_state.analysis_result = result
                        st.session_state.report_paths = result.get('reports', {})
                        
                        st.success("✅ Analysis complete!")
                        st.rerun()
        
        else:
            st.info("👆 Please upload a presentation file to begin analysis")
    
    else:
        # Display results
        result = st.session_state.analysis_result
        
        if result:
            display_results(result)
            
            # Download section
            st.markdown("---")
            st.markdown("### 📥 Download Reports")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            report_paths = st.session_state.report_paths
            
            with col1:
                if report_paths.get('pdf'):
                    pdf_path = report_paths['pdf']
                    if Path(pdf_path).exists():
                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=f.read(),
                                file_name=Path(pdf_path).name,
                                mime="application/pdf",
                                use_container_width=True
                            )
                    else:
                        st.info("PDF report path not found")
            
            with col2:
                if report_paths.get('ppt'):
                    ppt_path = report_paths['ppt']
                    if Path(ppt_path).exists():
                        with open(ppt_path, 'rb') as f:
                            st.download_button(
                                label="📊 Download PPT Summary",
                                data=f.read(),
                                file_name=Path(ppt_path).name,
                                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                use_container_width=True
                            )
                    else:
                        st.info("PPT report path not found")
            
            with col3:
                # JSON export
                if st.button("💾 Export JSON Data", use_container_width=True):
                    st.download_button(
                        label="📋 Download JSON",
                        data=json.dumps(result, indent=2),
                        file_name=f"analysis_{result.get('presentation_name', 'result')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            # Reset button
            st.markdown("---")
            if st.button("🔄 Analyze Another Presentation", use_container_width=True):
                st.session_state.analysis_complete = False
                st.session_state.analysis_result = None
                st.session_state.report_paths = None
                st.rerun()


if __name__ == "__main__":
    main()
