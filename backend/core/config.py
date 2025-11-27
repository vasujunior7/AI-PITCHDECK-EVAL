"""
⚙️ Configuration Management
Centralized settings using Pydantic
"""

from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Keys
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OPENAI_API_KEY: str = ""
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS into list"""
        if isinstance(self.ALLOWED_ORIGINS, str):
            return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
        return self.ALLOWED_ORIGINS
    
    # File Storage
    UPLOAD_DIR: str = "uploads"
    REPORTS_DIR: str = "reports"
    MAX_FILE_SIZE: int = 50  # MB
    ALLOWED_EXTENSIONS: List[str] = [".pptx", ".pdf"]
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # AI Models
    SBERT_MODEL: str = "all-MiniLM-L6-v2"
    BERT_MODEL: str = "bert-base-uncased"
    
    # Scoring Weights - EXTREME CONTENT-FIRST FRAMEWORK
    # CONTENT QUALITY (60% total) - DEPTH IS KING
    WEIGHT_DEPTH: float = 0.30          # PRIMARY METRIC - Content quality, semantic depth, AI analysis
    WEIGHT_COHERENCE: float = 0.10      # Logical reasoning
    WEIGHT_BLOOMS: float = 0.05         # Cognitive complexity
    WEIGHT_INNOVATION: float = 0.15     # Innovation, problem-solving, technical depth
    
    # INDUSTRY METRICS (38% total) - Market-ready evaluation
    WEIGHT_MARKET_RELEVANCE: float = 0.10         # Market fit, value proposition
    WEIGHT_EXECUTION_FEASIBILITY: float = 0.10    # Roadmap, resources, risk
    WEIGHT_DATA_EVIDENCE: float = 0.10            # Statistics, research, validation
    WEIGHT_IMPACT_SCALABILITY: float = 0.05       # Social impact, growth potential
    WEIGHT_PROFESSIONAL_QUALITY: float = 0.03     # Storytelling, engagement
    
    # PRESENTATION QUALITY (2% total) - BARELY MATTERS
    WEIGHT_CLARITY: float = 0.005       # Almost nothing
    WEIGHT_STRUCTURE: float = 0.005     # Almost nothing
    WEIGHT_DESIGN: float = 0.005        # Almost nothing
    WEIGHT_READABILITY: float = 0.005   # Almost nothing
    
    # Penalties
    WEIGHT_REDUNDANCY_PENALTY: float = 0.10  # Penalty for repetition
    
    # AI blend weight: MAXIMUM for semantic understanding (70% AI, 30% metrics)
    AI_BLEND: float = 0.70  # AI IS THE JUDGE - Content quality matters most!
    
    # PROFESSIONAL/TECHNICAL PRESENTATION THRESHOLDS
    # Base scores start at 40-50 range (not 0-30)
    # Design Requirements (UI/UX)
    DESIGN_MIN_THRESHOLD: float = 40.0          # Base acceptable (was 50)
    DESIGN_CRITICAL_THRESHOLD: float = 25.0     # Below this = poor (was 30)
    DESIGN_PENALTY_POINTS: float = 5.0          # Reduced penalty (was 8)
    DESIGN_CRITICAL_PENALTY: float = 10.0       # Reduced penalty (was 15)
    
    # Readability Requirements - PROFESSIONAL LEVEL (technical complexity is GOOD)
    READABILITY_MIN_THRESHOLD: float = 40.0     # Base for professional (was 45)
    READABILITY_CRITICAL_THRESHOLD: float = 20.0 # Even technical is OK (was 25)
    READABILITY_PENALTY_POINTS: float = 3.0     # Small penalty (was 8)
    READABILITY_CRITICAL_PENALTY: float = 5.0   # Small penalty (was 15)
    
    # Content Depth Requirements
    DEPTH_MIN_THRESHOLD: float = 45.0           # Base for professional (was 50)
    DEPTH_CRITICAL_THRESHOLD: float = 25.0      # Lowered threshold (was 30)
    DEPTH_PENALTY_POINTS: float = 5.0           # Reduced (was 8)
    DEPTH_CRITICAL_PENALTY: float = 8.0         # Reduced (was 12)
    
    # Structure Requirements
    STRUCTURE_MIN_THRESHOLD: float = 45.0       # Base for professional (was 55)
    STRUCTURE_PENALTY_POINTS: float = 5.0       # Reduced (was 8)
    
    # Multi-Component Failure Multiplier
    MULTI_FAILURE_MULTIPLIER: float = 0.90      # Reduce score by 10% (was 15%)
    
    # Visual Requirements (for academic presentations)
    MIN_IMAGES_PER_SLIDE: float = 0.3           # 30% of slides should have visuals (was 40%)
    NO_VISUAL_PENALTY: float = 5.0              # Smaller penalty
    
    # Thresholds
    REDUNDANCY_THRESHOLD: float = 0.85
    CROWDED_WORD_THRESHOLD: int = 80
    SPARSE_WORD_THRESHOLD: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from .env file


# Global settings instance
settings = Settings()
