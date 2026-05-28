"""
🗄️ Database Management System
SQLite database operations for candidate storage, analytics, and data persistence.

Features:
- SQLite database operations
- Candidate data storage and retrieval
- Ranking history tracking
- Search analytics
- Data export capabilities
- Database backup and restore
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import os
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager for resume screening system data persistence
    """
    
    def __init__(self, db_path: str = "data/resume_screening.db"):
        """Initialize database manager"""
        self.db_path = db_path
        self.connection = None
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            
            self._create_tables()
            logger.info(f"✅ Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            cursor = self.connection.cursor()
            
            # Candidates table - stores parsed resume data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    filename TEXT,
                    original_language TEXT,
                    skills TEXT,  -- JSON array
                    technical_skills TEXT,  -- JSON array
                    soft_skills TEXT,  -- JSON array
                    experience_text TEXT,
                    total_years INTEGER DEFAULT 0,
                    recent_roles TEXT,  -- JSON array
                    education_text TEXT,
                    degrees TEXT,  -- JSON array
                    location TEXT,
                    linkedin TEXT,
                    github TEXT,
                    certifications TEXT,  -- JSON array
                    projects TEXT,  -- JSON array
                    languages TEXT,  -- JSON array
                    completeness_score INTEGER DEFAULT 0,
                    raw_text TEXT,
                    processed_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Rankings table - stores ranking results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rankings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER,
                    job_description TEXT,
                    job_role TEXT,
                    final_score REAL,
                    rank_position INTEGER,
                    skills_match_score REAL,
                    experience_score REAL,
                    education_score REAL,
                    semantic_similarity_score REAL,
                    completeness_score REAL,
                    matched_skills TEXT,  -- JSON array
                    missing_skills TEXT,  -- JSON array
                    strengths TEXT,  -- JSON array
                    recommendations TEXT,  -- JSON array
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (candidate_id) REFERENCES candidates (id)
                )
            """)
            
            # Search queries table - stores search analytics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    results_count INTEGER DEFAULT 0,
                    search_type TEXT DEFAULT 'general',
                    execution_time REAL,
                    user_session TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Job descriptions table - stores frequently used job descriptions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_descriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    required_skills TEXT,  -- JSON array
                    experience_level INTEGER DEFAULT 0,
                    role_type TEXT,
                    seniority_level TEXT,
                    usage_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Application settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidates_email ON candidates(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidates_created_at ON candidates(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rankings_candidate_id ON rankings(candidate_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rankings_score ON rankings(final_score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_queries_created_at ON search_queries(created_at)")
            
            self.connection.commit()
            logger.info("✅ Database tables created/verified")
            
        except Exception as e:
            logger.error(f"❌ Error creating database tables: {e}")
    
    def add_candidate(self, candidate_data: Dict) -> Optional[int]:
        """
        Add a new candidate to the database
        
        Args:
            candidate_data: Dictionary containing candidate information
            
        Returns:
            int: Candidate ID if successful, None otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Prepare data with JSON serialization for lists
            insert_data = {
                'name': candidate_data.get('name', ''),
                'email': candidate_data.get('email', ''),
                'phone': candidate_data.get('phone', ''),
                'filename': candidate_data.get('filename', ''),
                'original_language': candidate_data.get('original_language', 'en'),
                'skills': json.dumps(candidate_data.get('skills', [])),
                'technical_skills': json.dumps(candidate_data.get('technical_skills', [])),
                'soft_skills': json.dumps(candidate_data.get('soft_skills', [])),
                'experience_text': candidate_data.get('experience', ''),
                'total_years': candidate_data.get('total_years', 0),
                'recent_roles': json.dumps(candidate_data.get('recent_roles', [])),
                'education_text': candidate_data.get('education', ''),
                'degrees': json.dumps(candidate_data.get('degrees', [])),
                'location': candidate_data.get('location', ''),
                'linkedin': candidate_data.get('linkedin', ''),
                'github': candidate_data.get('github', ''),
                'certifications': json.dumps(candidate_data.get('certifications', [])),
                'projects': json.dumps(candidate_data.get('projects', [])),
                'languages': json.dumps(candidate_data.get('languages', [])),
                'completeness_score': candidate_data.get('completeness_score', 0),
                'raw_text': candidate_data.get('raw_text', ''),
                'processed_text': candidate_data.get('processed_text', ''),
                'updated_at': datetime.now().isoformat()
            }
            
            # Check if candidate already exists (by email or filename)
            existing_id = self._find_existing_candidate(
                insert_data['email'], 
                insert_data['filename']
            )
            
            if existing_id:
                # Update existing candidate
                return self._update_candidate(existing_id, insert_data)
            else:
                # Insert new candidate
                placeholders = ', '.join(['?' for _ in insert_data])
                columns = ', '.join(insert_data.keys())
                
                cursor.execute(
                    f"INSERT INTO candidates ({columns}) VALUES ({placeholders})",
                    list(insert_data.values())
                )
                
                self.connection.commit()
                candidate_id = cursor.lastrowid
                
                logger.info(f"✅ Added candidate: {insert_data['name']} (ID: {candidate_id})")
                return candidate_id
        
        except Exception as e:
            logger.error(f"❌ Error adding candidate: {e}")
            return None
    
    def _find_existing_candidate(self, email: str, filename: str) -> Optional[int]:
        """Find existing candidate by email or filename"""
        try:
            cursor = self.connection.cursor()
            
            if email:
                cursor.execute("SELECT id FROM candidates WHERE email = ?", (email,))
                result = cursor.fetchone()
                if result:
                    return result['id']
            
            if filename:
                cursor.execute("SELECT id FROM candidates WHERE filename = ?", (filename,))
                result = cursor.fetchone()
                if result:
                    return result['id']
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding existing candidate: {e}")
            return None
    
    def _update_candidate(self, candidate_id: int, update_data: Dict) -> int:
        """Update existing candidate"""
        try:
            cursor = self.connection.cursor()
            
            # Prepare update query
            set_clause = ', '.join([f"{key} = ?" for key in update_data.keys()])
            values = list(update_data.values()) + [candidate_id]
            
            cursor.execute(
                f"UPDATE candidates SET {set_clause} WHERE id = ?",
                values
            )
            
            self.connection.commit()
            logger.info(f"✅ Updated candidate ID: {candidate_id}")
            return candidate_id
            
        except Exception as e:
            logger.error(f"Error updating candidate: {e}")
            return candidate_id
    
    def get_candidate(self, candidate_id: int) -> Optional[Dict]:
        """
        Get candidate by ID
        
        Args:
            candidate_id: Candidate ID
            
        Returns:
            Dict: Candidate data or None
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
            
            result = cursor.fetchone()
            if result:
                return self._format_candidate_data(dict(result))
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting candidate {candidate_id}: {e}")
            return None
    
    def get_all_candidates(self, limit: int = None) -> List[Dict]:
        """
        Get all candidates from database
        
        Args:
            limit: Maximum number of candidates to return
            
        Returns:
            List[Dict]: List of candidate data
        """
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT * FROM candidates ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            candidates = []
            for row in results:
                candidate = self._format_candidate_data(dict(row))
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting all candidates: {e}")
            return []
    
    def _format_candidate_data(self, raw_data: Dict) -> Dict:
        """Format candidate data from database (deserialize JSON fields)"""
        try:
            # Deserialize JSON fields
            json_fields = [
                'skills', 'technical_skills', 'soft_skills', 'recent_roles',
                'degrees', 'certifications', 'projects', 'languages'
            ]
            
            for field in json_fields:
                if field in raw_data and raw_data[field]:
                    try:
                        raw_data[field] = json.loads(raw_data[field])
                    except json.JSONDecodeError:
                        raw_data[field] = []
                else:
                    raw_data[field] = []
            
            # Rename fields to match expected format
            formatted_data = {
                'id': raw_data.get('id'),
                'name': raw_data.get('name', ''),
                'email': raw_data.get('email', ''),
                'phone': raw_data.get('phone', ''),
                'filename': raw_data.get('filename', ''),
                'original_language': raw_data.get('original_language', 'en'),
                'skills': raw_data.get('skills', []),
                'technical_skills': raw_data.get('technical_skills', []),
                'soft_skills': raw_data.get('soft_skills', []),
                'experience': raw_data.get('experience_text', ''),
                'total_years': raw_data.get('total_years', 0),
                'recent_roles': raw_data.get('recent_roles', []),
                'education': raw_data.get('education_text', ''),
                'degrees': raw_data.get('degrees', []),
                'location': raw_data.get('location', ''),
                'linkedin': raw_data.get('linkedin', ''),
                'github': raw_data.get('github', ''),
                'certifications': raw_data.get('certifications', []),
                'projects': raw_data.get('projects', []),
                'languages': raw_data.get('languages', []),
                'completeness_score': raw_data.get('completeness_score', 0),
                'raw_text': raw_data.get('raw_text', ''),
                'processed_text': raw_data.get('processed_text', ''),
                'created_at': raw_data.get('created_at'),
                'updated_at': raw_data.get('updated_at')
            }
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Error formatting candidate data: {e}")
            return raw_data
    
    def save_ranking_results(self, candidate_rankings: List[Dict], job_description: str) -> bool:
        """
        Save ranking results to database
        
        Args:
            candidate_rankings: List of ranked candidates
            job_description: Job description used for ranking
            
        Returns:
            bool: Success status
        """
        try:
            cursor = self.connection.cursor()
            
            # Extract job role from first candidate if available
            job_role = candidate_rankings[0].get('job_role', 'General') if candidate_rankings else 'General'
            
            for candidate in candidate_rankings:
                # Find candidate ID by name/email
                candidate_id = self._find_candidate_id_by_info(
                    candidate.get('name'), 
                    candidate.get('email')
                )
                
                if not candidate_id:
                    continue
                
                ranking_data = {
                    'candidate_id': candidate_id,
                    'job_description': job_description,
                    'job_role': job_role,
                    'final_score': candidate.get('score', 0),
                    'rank_position': candidate.get('rank', 0),
                    'skills_match_score': candidate.get('score_breakdown', {}).get('skills_match_score', 0),
                    'experience_score': candidate.get('score_breakdown', {}).get('experience_level_score', 0),
                    'education_score': candidate.get('score_breakdown', {}).get('education_match_score', 0),
                    'semantic_similarity_score': candidate.get('score_breakdown', {}).get('semantic_similarity_score', 0),
                    'completeness_score': candidate.get('score_breakdown', {}).get('completeness_score', 0),
                    'matched_skills': json.dumps(candidate.get('matched_skills', [])),
                    'missing_skills': json.dumps(candidate.get('missing_skills', [])),
                    'strengths': json.dumps(candidate.get('strengths', [])),
                    'recommendations': json.dumps(candidate.get('recommendations', [])),
                    'confidence': candidate.get('confidence', 0.5)
                }
                
                placeholders = ', '.join(['?' for _ in ranking_data])
                columns = ', '.join(ranking_data.keys())
                
                cursor.execute(
                    f"INSERT INTO rankings ({columns}) VALUES ({placeholders})",
                    list(ranking_data.values())
                )
            
            self.connection.commit()
            logger.info(f"✅ Saved ranking results for {len(candidate_rankings)} candidates")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving ranking results: {e}")
            return False
    
    def _find_candidate_id_by_info(self, name: str, email: str) -> Optional[int]:
        """Find candidate ID by name or email"""
        try:
            cursor = self.connection.cursor()
            
            if email:
                cursor.execute("SELECT id FROM candidates WHERE email = ?", (email,))
                result = cursor.fetchone()
                if result:
                    return result['id']
            
            if name and name.lower() != 'unknown':
                cursor.execute("SELECT id FROM candidates WHERE name = ?", (name,))
                result = cursor.fetchone()
                if result:
                    return result['id']
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding candidate ID: {e}")
            return None
    
    def log_search_query(self, query: str, results_count: int, search_type: str = 'general', 
                        execution_time: float = 0.0, user_session: str = 'default') -> bool:
        """
        Log search query for analytics
        
        Args:
            query: Search query text
            results_count: Number of results returned
            search_type: Type of search (general, skills, role, etc.)
            execution_time: Query execution time in seconds
            user_session: User session ID
            
        Returns:
            bool: Success status
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO search_queries (query_text, results_count, search_type, execution_time, user_session)
                VALUES (?, ?, ?, ?, ?)
            """, (query, results_count, search_type, execution_time, user_session))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error logging search query: {e}")
            return False
    
    def get_search_analytics(self, days: int = 30) -> Dict:
        """
        Get search analytics for the last N days
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict: Search analytics data
        """
        try:
            cursor = self.connection.cursor()
            
            # Get total search count
            cursor.execute("""
                SELECT COUNT(*) as total_searches 
                FROM search_queries 
                WHERE created_at >= datetime('now', '-{} days')
            """.format(days))
            total_searches = cursor.fetchone()['total_searches']
            
            # Get top search queries
            cursor.execute("""
                SELECT query_text, COUNT(*) as frequency 
                FROM search_queries 
                WHERE created_at >= datetime('now', '-{} days')
                GROUP BY query_text 
                ORDER BY frequency DESC 
                LIMIT 10
            """.format(days))
            top_queries = [dict(row) for row in cursor.fetchall()]
            
            # Get search types breakdown
            cursor.execute("""
                SELECT search_type, COUNT(*) as count 
                FROM search_queries 
                WHERE created_at >= datetime('now', '-{} days')
                GROUP BY search_type
            """.format(days))
            search_types = [dict(row) for row in cursor.fetchall()]
            
            # Get average execution time
            cursor.execute("""
                SELECT AVG(execution_time) as avg_time 
                FROM search_queries 
                WHERE created_at >= datetime('now', '-{} days') AND execution_time > 0
            """.format(days))
            avg_time_result = cursor.fetchone()
            avg_execution_time = avg_time_result['avg_time'] if avg_time_result['avg_time'] else 0
            
            return {
                'total_searches': total_searches,
                'top_queries': top_queries,
                'search_types': search_types,
                'avg_execution_time': round(avg_execution_time, 3),
                'analysis_period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting search analytics: {e}")
            return {}
    
    def get_candidate_statistics(self) -> Dict:
        """Get candidate statistics"""
        try:
            cursor = self.connection.cursor()
            
            # Total candidates
            cursor.execute("SELECT COUNT(*) as total FROM candidates")
            total_candidates = cursor.fetchone()['total']
            
            # Candidates by completeness score ranges
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN completeness_score >= 80 THEN 'High (80-100)'
                        WHEN completeness_score >= 60 THEN 'Medium (60-79)'
                        WHEN completeness_score >= 40 THEN 'Low (40-59)'
                        ELSE 'Very Low (0-39)'
                    END as completeness_range,
                    COUNT(*) as count
                FROM candidates 
                GROUP BY completeness_range
            """)
            completeness_stats = [dict(row) for row in cursor.fetchall()]
            
            # Top skills
            cursor.execute("SELECT skills FROM candidates WHERE skills IS NOT NULL")
            all_skills = []
            for row in cursor.fetchall():
                try:
                    skills = json.loads(row['skills'])
                    all_skills.extend(skills)
                except:
                    continue
            
            # Count skill frequencies
            skill_counts = {}
            for skill in all_skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
            
            top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Recent additions (last 7 days)
            cursor.execute("""
                SELECT COUNT(*) as recent_count 
                FROM candidates 
                WHERE created_at >= datetime('now', '-7 days')
            """)
            recent_additions = cursor.fetchone()['recent_count']
            
            return {
                'total_candidates': total_candidates,
                'completeness_distribution': completeness_stats,
                'top_skills': [{'skill': skill, 'count': count} for skill, count in top_skills],
                'recent_additions_7days': recent_additions
            }
            
        except Exception as e:
            logger.error(f"Error getting candidate statistics: {e}")
            return {}
    
    def export_candidates_to_csv(self, file_path: str) -> bool:
        """
        Export candidates data to CSV file
        
        Args:
            file_path: Path to save CSV file
            
        Returns:
            bool: Success status
        """
        try:
            candidates = self.get_all_candidates()
            
            if not candidates:
                logger.warning("No candidates to export")
                return False
            
            # Prepare data for CSV
            csv_data = []
            for candidate in candidates:
                csv_row = {
                    'Name': candidate['name'],
                    'Email': candidate['email'],
                    'Phone': candidate['phone'],
                    'Location': candidate['location'],
                    'Skills': ', '.join(candidate['skills']),
                    'Experience_Years': candidate['total_years'],
                    'Education': candidate['education'],
                    'Completeness_Score': candidate['completeness_score'],
                    'Created_At': candidate['created_at']
                }
                csv_data.append(csv_row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            df.to_csv(file_path, index=False)
            
            logger.info(f"✅ Exported {len(candidates)} candidates to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting candidates to CSV: {e}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create database backup
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            bool: Success status
        """
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"✅ Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return False
    
    def get_database_info(self) -> Dict:
        """Get database information and statistics"""
        try:
            cursor = self.connection.cursor()
            
            # Get table counts
            tables = ['candidates', 'rankings', 'search_queries', 'job_descriptions']
            table_counts = {}
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                table_counts[table] = cursor.fetchone()['count']
            
            # Get database file size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            db_size_mb = round(db_size / (1024 * 1024), 2)
            
            return {
                'database_path': self.db_path,
                'database_size_mb': db_size_mb,
                'table_counts': table_counts,
                'total_records': sum(table_counts.values()),
                'connection_active': bool(self.connection)
            }
            
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                logger.info("✅ Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    def __del__(self):
        """Destructor - ensure connection is closed"""
        self.close()

# Utility functions for testing and development
def test_database():
    """Test database functionality"""
    print("🗄️ Database Management Test")
    print("=" * 30)
    
    db = DatabaseManager()
    info = db.get_database_info()
    
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n✅ Database manager initialized successfully!")
    return db

if __name__ == "__main__":
    test_database()