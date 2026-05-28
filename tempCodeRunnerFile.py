
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add project directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

# Import custom modules (will be created in subsequent steps)
try:
    from utils.resume_parser import ResumeParser
    from models.ranking_engine import RankingEngine
    from utils.semantic_search import SemanticSearch
    from utils.database import DatabaseManager
    from components.visualizations import create_visualizations
    from utils.export_utils import ExportManager
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all modules are properly installed.")

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS styling for modern UI"""
    try:
        with open("static/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback inline CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .candidate-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            background: white;
        }
        </style>
        """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'candidates' not in st.session_state:
        st.session_state.candidates = []
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    if 'rankings' not in st.session_state:
        st.session_state.rankings = []
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

def main_header():
    """Display the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🚀 AI-Powered Resume Screening & Ranking System
        </h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">
            Transform your recruitment process with intelligent candidate analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

def sidebar_navigation():
    """Create sidebar navigation and controls"""
    st.sidebar.title("🎛️ Control Panel")
    
    # App navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📤 Upload & Parse", "📊 Ranking Dashboard", "🔍 Semantic Search", "📈 Analytics", "⚙️ Settings"]
    )
    
    st.sidebar.markdown("---")
    
    # Quick stats
    st.sidebar.subheader("📋 Quick Stats")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Candidates", len(st.session_state.candidates))
    with col2:
        st.metric("Processed", len(st.session_state.rankings))
    
    # Clear data button
    if st.sidebar.button("🗑️ Clear All Data", type="secondary"):
        st.session_state.candidates = []
        st.session_state.rankings = []
        st.session_state.job_description = ""
        st.sidebar.success("Data cleared!")
        st.experimental_rerun()
    
    return page

def upload_and_parse_page():
    """Main page for uploading and parsing resumes"""
    st.header("📤 Upload & Parse Resumes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Resume Files")
        uploaded_files = st.file_uploader(
            "Choose resume files (PDF, TXT, DOCX)",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload multiple resumes for batch processing"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
            
            # Process files button
            if st.button("🔄 Process Resumes", type="primary"):
                process_resumes(uploaded_files)
    
    with col2:
        st.subheader("Job Description")
        job_desc = st.text_area(
            "Enter job requirements:",
            value=st.session_state.job_description,
            height=200,
            help="Describe the role, required skills, and qualifications"
        )
        st.session_state.job_description = job_desc
        
        # Language settings
        st.subheader("🌍 Language Settings")
        auto_translate = st.checkbox("Auto-translate non-English resumes", value=True)
        target_language = st.selectbox("Target language:", ["English", "Spanish", "French", "German"])

def process_resumes(uploaded_files):
    """Process uploaded resume files"""
    parser = ResumeParser()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name}...")
        
        # Parse resume (this will be implemented in resume_parser.py)
        try:
            candidate_data = parser.parse_file(file)
            st.session_state.candidates.append(candidate_data)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("✅ Processing complete!")
    st.success(f"Successfully processed {len(uploaded_files)} resumes")

def ranking_dashboard_page():
    """Display candidate ranking dashboard"""
    st.header("📊 AI-Powered Ranking Dashboard")
    
    if not st.session_state.candidates:
        st.warning("⚠️ No candidates uploaded yet. Please upload resumes first.")
        return
    
    if not st.session_state.job_description:
        st.warning("⚠️ Please provide a job description for accurate ranking.")
        return
    
    # Ranking controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("Ranking Parameters")
    with col2:
        if st.button("🚀 Generate Rankings", type="primary"):
            generate_rankings()
    with col3:
        if st.button("📄 Export Results"):
            export_rankings()
    
    # Display rankings if available
    if st.session_state.rankings:
        display_ranking_table()
        display_ranking_charts()

def generate_rankings():
    """Generate AI-powered candidate rankings"""
    if not st.session_state.candidates or not st.session_state.job_description:
        return
    
    ranking_engine = RankingEngine()
    
    with st.spinner("🤖 AI is analyzing candidates..."):
        rankings = ranking_engine.rank_candidates(
            st.session_state.candidates,
            st.session_state.job_description
        )
        st.session_state.rankings = rankings
    
    st.success("✅ Rankings generated successfully!")

def display_ranking_table():
    """Display candidate ranking table"""
    df = pd.DataFrame(st.session_state.rankings)
    
    st.subheader("🏆 Candidate Rankings")
    
    # Color-code rows based on score
    def color_score(val):
        if val >= 80:
            return 'background-color: #d4edda'  # Green
        elif val >= 60:
            return 'background-color: #fff3cd'  # Yellow
        else:
            return 'background-color: #f8d7da'  # Red
    
    styled_df = df.style.applymap(color_score, subset=['score'])
    st.dataframe(styled_df, use_container_width=True)

def display_ranking_charts():
    """Display ranking visualization charts"""
    if not st.session_state.rankings:
        return
    
    df = pd.DataFrame(st.session_state.rankings)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution chart
        fig = px.histogram(df, x='score', title='Score Distribution', 
                          color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top candidates chart
        top_candidates = df.head(10)
        fig = px.bar(top_candidates, x='score', y='name', 
                    title='Top 10 Candidates', orientation='h',
                    color_discrete_sequence=['#764ba2'])
        st.plotly_chart(fig, use_container_width=True)

def semantic_search_page():
    """Semantic search interface"""
    st.header("🔍 Semantic Search")
    
    if not st.session_state.candidates:
        st.warning("⚠️ No candidates available for search. Please upload resumes first.")
        return
    
    search_query = st.text_input(
        "🔍 Search for skills, roles, or keywords:",
        placeholder="e.g., Python Data Scientist, Machine Learning Engineer"
    )
    
    if search_query:
        search_engine = SemanticSearch()
        results = search_engine.search(search_query, st.session_state.candidates)
        
        st.subheader("Search Results")
        for result in results:
            with st.expander(f"{result['name']} - Similarity: {result['similarity']:.2%}"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write(f"**Email:** {result['email']}")
                    st.write(f"**Phone:** {result['phone']}")
                with col2:
                    st.write(f"**Skills:** {', '.join(result['skills'])}")
                    st.write(f"**Experience:** {result.get('experience', 'Not specified')}")

def analytics_page():
    """Analytics and insights page"""
    st.header("📈 Analytics & Insights")
    
    if not st.session_state.candidates:
        st.warning("⚠️ No data available for analytics.")
        return
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Candidates", len(st.session_state.candidates))
    with col2:
        avg_score = sum(r['score'] for r in st.session_state.rankings) / len(st.session_state.rankings) if st.session_state.rankings else 0
        st.metric("Average Score", f"{avg_score:.1f}")
    with col3:
        high_score = len([r for r in st.session_state.rankings if r['score'] >= 80])
        st.metric("High Scorers", high_score)
    with col4:
        st.metric("Processing Time", "2.3s")
    
    # Skills analysis would go here
    st.subheader("🛠️ Skills Analysis")
    st.info("Skills analysis visualization will be implemented with the visualizations component.")

def settings_page():
    """Application settings and configuration"""
    st.header("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 AI Models")
        model_options = ["sentence-transformers/all-MiniLM-L6-v2", "distilbert-base-uncased", "bert-base-uncased"]
        selected_model = st.selectbox("Semantic Model:", model_options)
        
        threshold = st.slider("Similarity Threshold:", 0.0, 1.0, 0.7, 0.05)
        
        st.subheader("🌍 Language Settings")
        st.checkbox("Enable auto-translation", value=True)
        st.selectbox("Default language:", ["English", "Spanish", "French", "German"])
    
    with col2:
        st.subheader("📊 Display Settings")
        st.slider("Max results per page:", 10, 100, 50, 10)
        st.selectbox("Chart theme:", ["default", "dark", "minimal"])
        
        st.subheader("💾 Data Management")
        if st.button("🗄️ Backup Database"):
            st.info("Database backup feature will be implemented.")
        if st.button("🔄 Reset to Defaults"):
            st.info("Settings reset feature will be implemented.")

def export_rankings():
    """Export ranking results"""
    if not st.session_state.rankings:
        st.warning("No rankings to export.")
        return
    
    export_manager = ExportManager()
    
    # This will be implemented in export_utils.py
    st.success("Export functionality will be implemented in the ExportManager class.")

def main():
    """Main application entry point"""
    # Initialize app
    load_css()
    initialize_session_state()
    
    # Display header
    main_header()
    
    # Navigation
    page = sidebar_navigation()
    
    # Route to appropriate page
    if page == "📤 Upload & Parse":
        upload_and_parse_page()
    elif page == "📊 Ranking Dashboard":
        ranking_dashboard_page()
    elif page == "🔍 Semantic Search":
        semantic_search_page()
    elif page == "📈 Analytics":
        analytics_page()
    elif page == "⚙️ Settings":
        settings_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
         AI Resume Screening System v1.0 | Built  using Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()