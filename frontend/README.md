# 🎯 AI Pitch Deck Evaluator - Frontend

Beautiful, modern Streamlit frontend for the AI Pitch Deck Evaluator.

## Features

✨ **Modern UI/UX**
- Gradient backgrounds and glassmorphism effects
- Responsive card-based layouts
- Smooth animations and transitions
- Color-coded metrics and grades

🚀 **Functionality**
- File upload for PPT/PPTX/PDF
- Real-time analysis with progress tracking
- Comprehensive results display
- Download reports (PDF, PPT, JSON)

📊 **Insights Display**
- Overall score with letter grade
- Component breakdown visualization
- AI-powered strengths & improvements
- Prioritized recommendations

## Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Start the FastAPI backend first:**
   ```bash
   # In the backend directory
   uvicorn main:app --reload
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the app:**
   - Open your browser to `http://localhost:8501`
   - Upload your presentation file
   - Click "Analyze Presentation"
   - View results and download reports

## Configuration

The app connects to the FastAPI backend at `http://localhost:8000` by default.

To change this, modify the `API_BASE_URL` variable in `app.py`:

```python
API_BASE_URL = "http://your-backend-url:port"
```

## Features Overview

### Upload Section
- Drag-and-drop file upload
- Support for PPT, PPTX, and PDF formats
- File size and type validation

### Analysis Display
- **Overall Score**: Large, color-coded grade badge
- **Key Metrics**: Total slides, confidence, recommendations count
- **Component Scores**: Visual breakdown of each component
- **AI Insights**: Strengths and improvements
- **Recommendations**: Prioritized list with impact indicators

### Download Options
- PDF Report: Comprehensive analysis report
- PPT Summary: PowerPoint summary (if generated)
- JSON Export: Raw analysis data

## Design Elements

- **Color Scheme**: Purple gradient (modern, professional)
- **Typography**: Clean, readable fonts
- **Layout**: Card-based, responsive design
- **Animations**: Smooth transitions and progress indicators
- **Icons**: Emoji-based for universal recognition

## Troubleshooting

**Cannot connect to backend:**
- Ensure FastAPI server is running on `http://localhost:8000`
- Check firewall settings
- Verify API endpoint configuration

**File upload errors:**
- Ensure file is in supported format (PPT, PPTX, PDF)
- Check file size is reasonable (< 50 MB)
- Verify file is not corrupted

**Report download issues:**
- Check that backend generated the reports successfully
- Verify file paths in the backend response
- Ensure backend has write permissions

## Support

For issues or questions, please check:
1. Backend logs for API errors
2. Browser console for frontend errors
3. Streamlit terminal output for runtime errors
