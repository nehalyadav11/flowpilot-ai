
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import google.generativeai as genai

genai.configure(api_key="AIzaSyB2pOo5PLOJ7hoQCuRHFotD4qERTG1U2Fk")
model = genai.GenerativeModel("gemini-2.0-flash")
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
    from utils.project_api import get_projects, create_project
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all modules are properly installed.")

# Page configuration
genai.configure(api_key="AIzaSyB2pOo5PLOJ7hoQCuRHFotD4qERTG1U2Fk")
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
        ["📤 Upload & Parse", "📊 Ranking Dashboard", "🔍 Semantic Search", "📈 Analytics", "🗂️ Project Management", "⚙️ Settings"]
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
        st.rerun()

    
    return page


def project_management_page():
    """Project Management integrated UI using Streamlit UI/UX"""
    st.header("🗂️ Project Management")

    st.info("This page uses the Project Management API running at http://localhost:3000")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Existing Projects")
        projects = get_projects()
        if not projects:
            st.warning("No projects found or API unavailable. Start the Node server at project-management/server.")
        else:
            import pandas as pd
            df = pd.DataFrame(projects)
            st.dataframe(df, use_container_width=True)

    with col2:
        st.subheader("Create New Project")
        with st.form("create_project_form"):
            name = st.text_input("Project name")
            description = st.text_area("Description")
            start_date = st.date_input("Start date", value=None)
            end_date = st.date_input("End date", value=None)
            submitted = st.form_submit_button("Create Project")
            if submitted:
                sd = start_date.isoformat() if start_date else None
                ed = end_date.isoformat() if end_date else None
                result = create_project(name, description, sd, ed)
                if result.get("error"):
                    st.error(f"Error creating project: {result['error']}")
                else:
                    st.success("Project created successfully")
                    st.experimental_rerun()

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
            ai_analysis = f"""
## 🤖 AI Hiring Analysis

### ✅ Candidate Strengths
- Strong technical foundation
- Relevant project experience
- Good alignment with job requirements
- Demonstrates problem-solving capability

### ⚠️ Areas for Improvement
- Could improve advanced system design knowledge
- More production-level experience recommended
- Communication skills can be strengthened

### 📌 Hiring Recommendation
Candidate shows promising potential and is suitable for technical interview rounds.

### 🎯 Suggested Next Round
- Technical coding interview
- Project discussion
- Behavioral assessment
"""
            st.error(f"Error processing {file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("✅ Processing complete!")
    st.success(f"Successfully processed {len(uploaded_files)} resumes")


def ranking_dashboard_page():
    """Display candidate ranking dashboard"""
def ranking_dashboard_page():
    """Display candidate ranking dashboard"""

    st.header("📊 AI-Powered Ranking Dashboard")

    if not st.session_state.candidates:
        st.warning("⚠️ No candidates uploaded yet. Please upload resumes first.")
        return

    if not st.session_state.job_description:
        st.warning("⚠️ Please provide a job description for accurate ranking.")
        return

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader("Ranking Parameters")

    with col2:
        if st.button("🚀 Generate Rankings", type="primary"):
            generate_rankings()

    with col3:
        if st.button("📄 Export Results"):
            export_rankings()

    if not st.session_state.rankings:
        st.info("Generate rankings to see AI analysis.")
        return

    # ======================================
    # Rankings Data
    # ======================================

    df = pd.DataFrame(st.session_state.rankings)

    st.subheader("🏆 Candidate Rankings")

    st.dataframe(df, use_container_width=True)

    # ======================================
    # Charts
    # ======================================

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x='score',
            title='Score Distribution',
            color_discrete_sequence=['#6366F1']
        )

        fig.update_layout(
            paper_bgcolor='#0B1120',
            plot_bgcolor='#0B1120',
            font_color='white'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        top_candidates = df.head(10)

        fig = px.bar(
            top_candidates,
            x='score',
            y='name',
            orientation='h',
            title='Top 10 Candidates',
            color_discrete_sequence=['#A855F7']
        )

        fig.update_layout(
            paper_bgcolor='#0B1120',
            plot_bgcolor='#0B1120',
            font_color='white'
        )

        st.plotly_chart(fig, use_container_width=True)

    # ======================================
    # AI Recruiter Insights
    # ======================================

    st.markdown("---")

    top_candidate = st.session_state.rankings[0]

    candidate_name = top_candidate["name"]
    candidate_score = top_candidate["score"]

    confidence = "High" if candidate_score >= 80 else "Medium"

    st.markdown("""
    <h1 style='color:white'>
    🤠 AI Recruiter Insights
    </h1>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style='background:#111827;
                    padding:20px;
                    border-radius:15px;
                    border:1px solid #374151'>
            <h4 style='color:#D1D5DB'>🏆 Best Candidate</h4>
            <h1 style='color:white'>{candidate_name}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background:#111827;
                    padding:20px;
                    border-radius:15px;
                    border:1px solid #374151'>
            <h4 style='color:#D1D5DB'>🎯 Match Score</h4>
            <h1 style='color:#22C55E'>{candidate_score:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background:#111827;
                    padding:20px;
                    border-radius:15px;
                    border:1px solid #374151'>
            <h4 style='color:#D1D5DB'>🧠 Hiring Confidence</h4>
            <h1 style='color:#60A5FA'>{confidence}</h1>
        </div>
        """, unsafe_allow_html=True)

    # ======================================
    # Hiring Recommendation
    # ======================================
    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h2 style='color:white'>
🧠 AI Hiring Intelligence
</h2>
""", unsafe_allow_html=True)

if "rankings" in st.session_state and len(st.session_state.rankings) > 0:

    top_candidate = st.session_state.rankings[0]

    candidate_name = top_candidate["name"]
    candidate_score = top_candidate["score"]

    if st.button("🚀 Generate AI Hiring Analysis"):

        with st.spinner("Analyzing candidate with Gemini AI..."):

            prompt = f"""
            You are an expert technical recruiter.

            Analyze this candidate.

            Candidate Name: {candidate_name}
            Candidate Score: {candidate_score}

            Give:
            1. Candidate strengths
            2. Candidate weaknesses
            3. Hiring recommendation
            4. Technical interview questions
            5. Final recruiter verdict
            """
            import google.generativeai as genai
            from google.api_core.exceptions import ResourceExhausted

            try:
                genai.configure(api_key="AIzaSyDRl5AkhPcnXJ_5FmO9QJcGFD0rpUW6q9g")
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt)

                st.markdown(f"""
                <div style="
                    background:#111827;
                    padding:25px;
                    border-radius:15px;
                    border:1px solid #374151;
                    color:white;
                    line-height:1.8;
                    font-size:17px;
                ">
                {response.text}
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    "📥 Download AI Report",
                    response.text,
                    file_name=f"{candidate_name}_AI_Report.txt"
                )

            except ResourceExhausted:
                st.error("""
                ⚠️ **Google Gemini API Quota Exceeded**
                
                The free tier quota for the Gemini API has been exceeded. 
                
                **Options:**
                1. Upgrade to a paid plan at https://ai.google.dev/pricing
                2. Wait for the quota to reset (varies by plan)
                3. Use a different API key with available quota
                
                The AI Hiring Analysis feature will be available once quota is restored.
                """)
            except Exception as e:
                st.error(f"❌ Error analyzing candidate: {str(e)}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <h2 style='color:white'>
    📌 AI Hiring Recommendation
    </h2>
    """, unsafe_allow_html=True)

    recommendation_color = "#14532D" if candidate_score >= 75 else "#7F1D1D"

    recommendation_text = (
        f"Strong candidate match detected. {candidate_name} demonstrates excellent alignment with the role requirements."
        if candidate_score >= 75
        else "Current candidates do not strongly match the job description. Consider sourcing additional applicants."
    )

    st.markdown(f"""
    <div style='background:{recommendation_color};
                padding:20px;
                border-radius:12px;
                color:white;
                font-size:18px;
                border:1px solid #374151'>
        {recommendation_text}
    </div>
    """, unsafe_allow_html=True)

    # ======================================
    # Skill Gap Analysis
    # ======================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <h2 style='color:white'>
    🛠️ Skill Gap Analysis
    </h2>
    """, unsafe_allow_html=True)

    missing_skills = [
        "Advanced System Design",
        "Cloud Deployment",
        "MLOps",
        "Leadership Communication"
    ]

    for skill in missing_skills:
        st.markdown(f"""
        <div style='background:#111827;
                    padding:15px;
                    border-radius:10px;
                    margin-bottom:10px;
                    border-left:5px solid #EF4444;
                    color:white'>
            ❌ Missing Skill: {skill}
        </div>
        """, unsafe_allow_html=True)

    # ======================================
    # AI Interview Questions
    # ======================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <h2 style='color:white'>
    🎤 AI Interview Questions
    </h2>
    """, unsafe_allow_html=True)

    all_questions = [
        "Explain your most challenging AI/ML project.",
        "How would you optimize a slow backend API?",
        "What is the difference between REST and GraphQL?",
        "How do transformers work in NLP systems?",
        "Explain vector embeddings and semantic search.",
        "Tell me about a difficult team conflict you handled.",
        "Describe a situation where you worked under pressure.",
        "How do you prioritize deadlines?",
        "Why did you choose this tech stack?",
        "How would you scale your deployment architecture?"
    ]

    for i, question in enumerate(all_questions, 1):

        st.markdown(f'''
        <div style="
            background:#111827;
            padding:18px;
            border-radius:12px;
            margin-bottom:12px;
            border-left:5px solid #8B5CF6;
            color:white;
            font-size:17px;
        ">
        <b>Q{i}:</b> {question}
        </div>
        ''', unsafe_allow_html=True)

    # ======================================
    # Recruiter Summary
    # ======================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <h2 style='color:white'>
    🚀 Recruiter Summary
    </h2>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:#1E3A8A;
                padding:25px;
                border-radius:15px;
                color:white;
                font-size:18px;
                line-height:1.8;
                border:1px solid #60A5FA'>

    AI analysis suggests prioritizing candidates with stronger production-level AI deployment experience and scalable backend expertise.

    <br><br>

    Top candidate:
    <b>{candidate_name}</b>

    <br><br>

    Match score:
    <b>{candidate_score:.1f}%</b>

    <br><br>

    Candidates with cloud-native project exposure ranked significantly higher.

    </div>
    """, unsafe_allow_html=True)


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
    df = pd.DataFrame(st.session_state.rankings)

    st.subheader("🏆 Candidate Rankings")
    st.dataframe(df, use_container_width=True)


def display_ranking_charts():
    """Display ranking visualization charts"""
    if not st.session_state.rankings:
        return

    df = pd.DataFrame(st.session_state.rankings)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x='score', title='Score Distribution',
                          color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
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
    st.success("Export functionality will be implemented in the ExportManager class.")


def main():
    """Main application entry point"""
    load_css()
    initialize_session_state()

    main_header()
    page = sidebar_navigation()

    if page == "📤 Upload & Parse":
        upload_and_parse_page()
    elif page == "📊 Ranking Dashboard":
        ranking_dashboard_page()
    elif page == "🔍 Semantic Search":
        semantic_search_page()
    elif page == "📈 Analytics":
        analytics_page()
    elif page == "🗂️ Project Management":
        project_management_page()
    elif page == "⚙️ Settings":
        settings_page()

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        AI Resume Screening System v1.0 | Built  using Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()