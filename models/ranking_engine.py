"""
🤖 AI Ranking Engine
Advanced candidate ranking system using semantic similarity and machine learning.

Features:
- Sentence-BERT embeddings for semantic matching
- Multi-criteria scoring algorithm
- Skills matching with weighted importance
- Experience level assessment
- Job description analysis
- Real-time ranking with confidence scores
"""

import numpy as np
import pandas as pd
import re
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

# ML and NLP libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import torch
except ImportError:
    print("ML libraries not available. Install: pip install sentence-transformers scikit-learn torch")

try:
    import spacy
except ImportError:
    print("spaCy not available. Install: pip install spacy")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RankingEngine:
    """
    AI-powered resume ranking engine with semantic similarity matching
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the ranking engine with AI models"""
        self.model_name = model_name
        self.sentence_model = None
        self.nlp = None
        self.tfidf_vectorizer = None
        
        # Scoring weights for different criteria
        self.scoring_weights = {
            'skills_match': 0.35,           # 35% - Most important
            'experience_level': 0.25,       # 25% - Very important  
            'education_match': 0.15,        # 15% - Moderately important
            'semantic_similarity': 0.20,    # 20% - Context understanding
            'completeness_bonus': 0.05      # 5% - Profile completeness
        }
        
        # Job role to skills mapping for enhanced matching
        self.role_skill_mappings = {
            'data scientist': ['python', 'machine learning', 'statistics', 'pandas', 'numpy', 'sql', 'tensorflow', 'pytorch'],
            'software engineer': ['python', 'java', 'javascript', 'git', 'algorithms', 'data structures', 'sql'],
            'web developer': ['html', 'css', 'javascript', 'react', 'node.js', 'php', 'mysql', 'git'],
            'devops engineer': ['aws', 'docker', 'kubernetes', 'jenkins', 'terraform', 'linux', 'python', 'git'],
            'machine learning engineer': ['python', 'tensorflow', 'pytorch', 'scikit-learn', 'docker', 'aws', 'git'],
            'mobile developer': ['swift', 'kotlin', 'react native', 'flutter', 'ios', 'android', 'git'],
            'product manager': ['analytics', 'sql', 'project management', 'agile', 'jira', 'communication'],
            'data analyst': ['sql', 'python', 'excel', 'tableau', 'powerbi', 'statistics', 'pandas']
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models and components"""
        try:
            # Initialize Sentence Transformer
            self.sentence_model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Sentence Transformer {self.model_name} loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Sentence Transformer: {e}")
        
        try:
            # Initialize spaCy model for NLP processing
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded")
        except Exception as e:
            logger.warning(f"⚠️ spaCy model not available: {e}")
        
        try:
            # Initialize TF-IDF vectorizer for keyword matching
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            logger.info("✅ TF-IDF vectorizer initialized")
        except Exception as e:
            logger.warning(f"⚠️ TF-IDF vectorizer initialization failed: {e}")
    
    def rank_candidates(self, candidates: List[Dict], job_description: str) -> List[Dict]:
        """
        Rank candidates based on job description using multi-criteria scoring
        
        Args:
            candidates: List of candidate dictionaries
            job_description: Job requirements and description
            
        Returns:
            List[Dict]: Ranked candidates with scores and insights
        """
        if not candidates:
            logger.warning("No candidates to rank")
            return []
        
        if not job_description or len(job_description.strip()) < 20:
            logger.warning("Job description too short for accurate ranking")
            return self._fallback_ranking(candidates)
        
        try:
            # Analyze job description
            job_analysis = self._analyze_job_description(job_description)
            logger.info(f"Job analysis completed: {len(job_analysis['required_skills'])} skills identified")
            
            # Calculate scores for each candidate
            ranked_candidates = []
            for candidate in candidates:
                if not self._is_valid_candidate(candidate):
                    continue
                
                # Calculate individual scores
                scores = self._calculate_candidate_scores(candidate, job_analysis)
                
                # Calculate weighted final score
                final_score = self._calculate_weighted_score(scores)
                
                # Generate insights
                insights = self._generate_insights(candidate, job_analysis, scores)
                
                # Prepare ranked candidate data
                ranked_candidate = {
                    'name': candidate.get('name', 'Unknown'),
                    'email': candidate.get('email', ''),
                    'phone': candidate.get('phone', ''),
                    'score': round(final_score, 1),
                    'skills': candidate.get('skills', []),
                    'experience': candidate.get('total_years', 0),
                    'filename': candidate.get('filename', ''),
                    'matched_skills': insights['matched_skills'],
                    'missing_skills': insights['missing_skills'],
                    'strengths': insights['strengths'],
                    'recommendations': insights['recommendations'],
                    'score_breakdown': scores,
                    'confidence': self._calculate_confidence(scores),
                    'ranking_date': datetime.now().isoformat()
                }
                
                ranked_candidates.append(ranked_candidate)
            
            # Sort by score (descending)
            ranked_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # Add ranking positions
            for i, candidate in enumerate(ranked_candidates):
                candidate['rank'] = i + 1
            
            logger.info(f"✅ Successfully ranked {len(ranked_candidates)} candidates")
            return ranked_candidates
            
        except Exception as e:
            logger.error(f"❌ Error in candidate ranking: {e}")
            return self._fallback_ranking(candidates)
    
    def _analyze_job_description(self, job_description: str) -> Dict:
        """Analyze job description to extract requirements"""
        analysis = {
            'required_skills': [],
            'preferred_skills': [],
            'experience_level': 0,
            'education_requirements': [],
            'role_type': '',
            'seniority_level': 'mid',
            'key_phrases': [],
            'embeddings': None
        }
        
        try:
            text_lower = job_description.lower()
            
            # Extract required skills using patterns and NLP
            analysis['required_skills'] = self._extract_job_skills(job_description)
            
            # Determine role type
            analysis['role_type'] = self._identify_role_type(job_description)
            
            # Extract experience requirements
            analysis['experience_level'] = self._extract_experience_requirement(job_description)
            
            # Determine seniority level
            analysis['seniority_level'] = self._identify_seniority_level(job_description)
            
            # Extract education requirements
            analysis['education_requirements'] = self._extract_education_requirements(job_description)
            
            # Generate embeddings for semantic matching
            if self.sentence_model:
                analysis['embeddings'] = self.sentence_model.encode([job_description])
            
            # Extract key phrases using TF-IDF
            if self.tfidf_vectorizer:
                try:
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform([job_description])
                    feature_names = self.tfidf_vectorizer.get_feature_names_out()
                    scores = tfidf_matrix.toarray()[0]
                    
                    # Get top phrases
                    top_indices = scores.argsort()[-10:][::-1]
                    analysis['key_phrases'] = [feature_names[i] for i in top_indices if scores[i] > 0]
                except:
                    pass
            
            return analysis
            
        except Exception as e:
            logger.error(f"Job analysis failed: {e}")
            return analysis
    
    def _extract_job_skills(self, job_description: str) -> List[str]:
        """Extract required skills from job description"""
        skills = set()
        text_lower = job_description.lower()
        
        # Check against all known skills from role mappings
        all_known_skills = set()
        for skill_list in self.role_skill_mappings.values():
            all_known_skills.update(skill_list)
        
        # Add common additional skills
        additional_skills = [
            'communication', 'teamwork', 'problem solving', 'analytical thinking',
            'project management', 'leadership', 'agile', 'scrum', 'rest api',
            'microservices', 'ci/cd', 'testing', 'debugging', 'optimization'
        ]
        all_known_skills.update(additional_skills)
        
        # Find mentioned skills
        for skill in all_known_skills:
            # Look for exact matches and variations
            patterns = [
                rf'\b{re.escape(skill)}\b',
                rf'\b{re.escape(skill)}\.js\b',
                rf'\b{re.escape(skill)}\.py\b',
                rf'\b{re.escape(skill)}/\b'
            ]
            
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    skills.add(skill)
                    break
        
        # Extract skills from common sections
        skill_sections = re.findall(
            r'(?:requirements?|skills?|technologies?|experience)[:\-\s]+(.*?)(?:\n\n|\n[A-Z]|\Z)',
            job_description, re.IGNORECASE | re.DOTALL
        )
        
        for section in skill_sections:
            section_skills = re.split(r'[,;•\n\t]+', section)
            for skill in section_skills:
                skill = skill.strip().lower()
                if skill and 2 <= len(skill) <= 25 and not skill.isdigit():
                    # Filter out common non-skills
                    if not any(word in skill for word in ['years', 'experience', 'degree', 'bachelor', 'master']):
                        skills.add(skill)
        
        return list(skills)[:15]  # Limit to top 15 skills
    
    def _identify_role_type(self, job_description: str) -> str:
        """Identify the primary role type from job description"""
        text_lower = job_description.lower()
        
        # Check for role keywords in order of specificity
        role_keywords = {
            'data scientist': ['data scientist', 'data science'],
            'machine learning engineer': ['machine learning engineer', 'ml engineer', 'ai engineer'],
            'data analyst': ['data analyst', 'business analyst', 'analytics'],
            'devops engineer': ['devops', 'platform engineer', 'infrastructure engineer'],
            'mobile developer': ['mobile developer', 'ios developer', 'android developer'],
            'web developer': ['web developer', 'frontend developer', 'backend developer', 'full stack'],
            'software engineer': ['software engineer', 'software developer', 'developer'],
            'product manager': ['product manager', 'product owner'],
        }
        
        for role, keywords in role_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return role
        
        return 'software engineer'  # Default fallback
    
    def _extract_experience_requirement(self, job_description: str) -> int:
        """Extract required years of experience"""
        # Look for experience patterns
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+?\s*years?\s*(?:professional\s*)?experience',
            r'minimum\s*(\d+)\+?\s*years?',
            r'at least\s*(\d+)\+?\s*years?'
        ]
        
        experience_years = []
        for pattern in patterns:
            matches = re.findall(pattern, job_description.lower())
            experience_years.extend([int(match) for match in matches])
        
        return max(experience_years) if experience_years else 2  # Default to 2 years
    
    def _identify_seniority_level(self, job_description: str) -> str:
        """Identify seniority level from job description"""
        text_lower = job_description.lower()
        
        if any(keyword in text_lower for keyword in ['senior', 'lead', 'principal', 'staff', 'architect']):
            return 'senior'
        elif any(keyword in text_lower for keyword in ['junior', 'entry', 'associate', 'intern']):
            return 'junior'
        else:
            return 'mid'
    
    def _extract_education_requirements(self, job_description: str) -> List[str]:
        """Extract education requirements"""
        education_requirements = []
        text_lower = job_description.lower()
        
        degree_patterns = [
            r'\b(bachelor|master|phd|doctorate|mba)\b',
            r'\b(bs|ms|ba|ma|btech|mtech)\b',
            r'\b(degree)\b'
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text_lower)
            education_requirements.extend(matches)
        
        return list(set(education_requirements))
    
    def _calculate_candidate_scores(self, candidate: Dict, job_analysis: Dict) -> Dict:
        """Calculate individual scores for different criteria"""
        scores = {
            'skills_match_score': 0.0,
            'experience_level_score': 0.0,
            'education_match_score': 0.0,
            'semantic_similarity_score': 0.0,
            'completeness_score': 0.0
        }
        
        try:
            # 1. Skills matching score
            scores['skills_match_score'] = self._calculate_skills_match_score(
                candidate.get('skills', []), job_analysis['required_skills']
            )
            
            # 2. Experience level score
            scores['experience_level_score'] = self._calculate_experience_score(
                candidate.get('total_years', 0), job_analysis['experience_level']
            )
            
            # 3. Education matching score
            scores['education_match_score'] = self._calculate_education_score(
                candidate.get('degrees', []), job_analysis['education_requirements']
            )
            
            # 4. Semantic similarity score
            scores['semantic_similarity_score'] = self._calculate_semantic_similarity(
                candidate, job_analysis
            )
            
            # 5. Profile completeness score
            scores['completeness_score'] = candidate.get('completeness_score', 0) / 100.0
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating candidate scores: {e}")
            return scores
    
    def _calculate_skills_match_score(self, candidate_skills: List[str], required_skills: List[str]) -> float:
        """Calculate skills matching score"""
        if not candidate_skills or not required_skills:
            return 0.0
        
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        required_skills_lower = [skill.lower() for skill in required_skills]
        
        # Direct matches
        direct_matches = len(set(candidate_skills_lower) & set(required_skills_lower))
        
        # Semantic matches using embeddings
        semantic_matches = 0
        if self.sentence_model and len(candidate_skills) > 0 and len(required_skills) > 0:
            try:
                candidate_embeddings = self.sentence_model.encode(candidate_skills)
                required_embeddings = self.sentence_model.encode(required_skills)
                
                similarity_matrix = cosine_similarity(candidate_embeddings, required_embeddings)
                
                # Count matches above threshold (0.7)
                for i in range(len(candidate_skills)):
                    if np.max(similarity_matrix[i]) > 0.7:
                        semantic_matches += 1
                        
            except Exception as e:
                logger.warning(f"Semantic skills matching failed: {e}")
        
        # Combine direct and semantic matches
        total_matches = max(direct_matches, semantic_matches)
        match_ratio = total_matches / len(required_skills)
        
        # Apply scoring curve (reward high matches more)
        if match_ratio >= 0.8:
            return 1.0
        elif match_ratio >= 0.6:
            return 0.8
        elif match_ratio >= 0.4:
            return 0.6
        elif match_ratio >= 0.2:
            return 0.4
        else:
            return match_ratio * 2  # Scale up low matches
    
    def _calculate_experience_score(self, candidate_years: int, required_years: int) -> float:
        """Calculate experience matching score"""
        if candidate_years >= required_years:
            # Exact or over-qualified
            if candidate_years <= required_years * 1.5:
                return 1.0  # Perfect match
            else:
                # Slightly over-qualified (diminishing returns)
                return max(0.8, 1.0 - (candidate_years - required_years * 1.5) * 0.05)
        else:
            # Under-qualified
            ratio = candidate_years / max(required_years, 1)
            return ratio * 0.8  # Cap at 0.8 for under-qualified
    
    def _calculate_education_score(self, candidate_degrees: List[str], required_education: List[str]) -> float:
        """Calculate education matching score"""
        if not required_education:
            return 1.0  # No requirements = full score
        
        if not candidate_degrees:
            return 0.3  # Some points for having a profile even without degree info
        
        candidate_degrees_lower = [degree.lower() for degree in candidate_degrees]
        required_education_lower = [req.lower() for req in required_education]
        
        # Check for matches
        matches = 0
        for req in required_education_lower:
            for degree in candidate_degrees_lower:
                if req in degree or degree in req:
                    matches += 1
                    break
        
        if matches > 0:
            return min(1.0, matches / len(required_education) + 0.2)  # Bonus for any match
        else:
            return 0.5  # Neutral score if no specific match but has education
    
    def _calculate_semantic_similarity(self, candidate: Dict, job_analysis: Dict) -> float:
        """Calculate semantic similarity between candidate profile and job description"""
        if not self.sentence_model or not job_analysis.get('embeddings') is not None:
            return 0.5  # Neutral score if no embeddings
        
        try:
            # Create candidate text representation
            candidate_text = self._create_candidate_text_representation(candidate)
            
            if not candidate_text:
                return 0.0
            
            # Generate candidate embeddings
            candidate_embedding = self.sentence_model.encode([candidate_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(candidate_embedding, job_analysis['embeddings'])[0][0]
            
            # Normalize to 0-1 range and apply sigmoid for better distribution
            normalized_similarity = max(0, min(1, similarity))
            return normalized_similarity
            
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.5
    
    def _create_candidate_text_representation(self, candidate: Dict) -> str:
        """Create a text representation of candidate for semantic analysis"""
        text_parts = []
        
        # Add skills
        skills = candidate.get('skills', [])
        if skills:
            text_parts.append(f"Skills: {', '.join(skills)}")
        
        # Add experience
        experience = candidate.get('experience', '')
        if experience:
            text_parts.append(f"Experience: {experience}")
        
        # Add education
        education = candidate.get('education', '')
        if education:
            text_parts.append(f"Education: {education}")
        
        # Add roles
        roles = candidate.get('recent_roles', [])
        if roles:
            text_parts.append(f"Recent roles: {', '.join(roles)}")
        
        return ' '.join(text_parts)
    
    def _calculate_weighted_score(self, scores: Dict) -> float:
        """Calculate final weighted score"""
        final_score = 0.0
        
        for criterion, weight in self.scoring_weights.items():
            if criterion == 'completeness_bonus':
                score_key = 'completeness_score'
            else:
                score_key = f"{criterion}_score"
            
            score = scores.get(score_key, 0.0)
            final_score += score * weight
        
        return min(100.0, final_score * 100)  # Convert to 0-100 scale
    
    def _generate_insights(self, candidate: Dict, job_analysis: Dict, scores: Dict) -> Dict:
        """Generate insights and recommendations for the candidate"""
        insights = {
            'matched_skills': [],
            'missing_skills': [],
            'strengths': [],
            'recommendations': []
        }
        
        try:
            candidate_skills = [skill.lower() for skill in candidate.get('skills', [])]
            required_skills = [skill.lower() for skill in job_analysis['required_skills']]
            
            # Find matched skills
            insights['matched_skills'] = [
                skill for skill in candidate.get('skills', [])
                if skill.lower() in required_skills
            ]
            
            # Find missing skills
            insights['missing_skills'] = [
                skill for skill in job_analysis['required_skills']
                if skill.lower() not in candidate_skills
            ][:5]  # Limit to top 5
            
            # Generate strengths
            if scores['skills_match_score'] > 0.8:
                insights['strengths'].append("Excellent technical skills match")
            if scores['experience_level_score'] > 0.8:
                insights['strengths'].append("Strong experience level")
            if scores['semantic_similarity_score'] > 0.7:
                insights['strengths'].append("Good overall profile alignment")
            if scores['completeness_score'] > 0.8:
                insights['strengths'].append("Comprehensive resume")
            
            # Generate recommendations
            if scores['skills_match_score'] < 0.6:
                insights['recommendations'].append("Consider skills assessment or training")
            if scores['experience_level_score'] < 0.5:
                insights['recommendations'].append("May need mentoring or junior role")
            if len(insights['missing_skills']) > 3:
                insights['recommendations'].append("Focus on key missing technical skills")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return insights
    
    def _calculate_confidence(self, scores: Dict) -> float:
        """Calculate confidence level of the ranking"""
        # Base confidence on score variance and completeness
        score_values = [score for score in scores.values() if isinstance(score, (int, float))]
        
        if not score_values:
            return 0.5
        
        # Higher variance = lower confidence
        variance = np.var(score_values)
        completeness = scores.get('completeness_score', 0.5)
        
        # Combine factors
        confidence = min(1.0, (1 - variance) * completeness + 0.3)
        return round(confidence, 2)
    
    def _is_valid_candidate(self, candidate: Dict) -> bool:
        """Check if candidate has minimum required information for ranking"""
        required_fields = ['name', 'skills']
        return all(candidate.get(field) for field in required_fields)
    
    def _fallback_ranking(self, candidates: List[Dict]) -> List[Dict]:
        """Fallback ranking when main algorithm fails"""
        logger.info("Using fallback ranking based on completeness scores")
        
        fallback_candidates = []
        for i, candidate in enumerate(candidates):
            if not self._is_valid_candidate(candidate):
                continue
            
            # Simple scoring based on available information
            score = candidate.get('completeness_score', 50)
            
            # Bonus for more skills
            skills_count = len(candidate.get('skills', []))
            score += min(20, skills_count * 2)
            
            # Bonus for experience
            experience = candidate.get('total_years', 0)
            score += min(15, experience * 2)
            
            fallback_candidate = {
                'name': candidate.get('name', 'Unknown'),
                'email': candidate.get('email', ''),
                'phone': candidate.get('phone', ''),
                'score': min(100, score),
                'skills': candidate.get('skills', []),
                'experience': candidate.get('total_years', 0),
                'filename': candidate.get('filename', ''),
                'rank': i + 1,
                'matched_skills': candidate.get('skills', [])[:3],  # Show first 3 skills
                'missing_skills': [],
                'strengths': ['Profile available'],
                'recommendations': ['Complete technical assessment'],
                'confidence': 0.5,
                'ranking_date': datetime.now().isoformat()
            }
            
            fallback_candidates.append(fallback_candidate)
        
        # Sort by score
        fallback_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Update ranks
        for i, candidate in enumerate(fallback_candidates):
            candidate['rank'] = i + 1
        
        return fallback_candidates
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models and configuration"""
        return {
            'sentence_transformer_model': self.model_name,
            'sentence_transformer_loaded': bool(self.sentence_model),
            'spacy_model_loaded': bool(self.nlp),
            'tfidf_vectorizer_loaded': bool(self.tfidf_vectorizer),
            'scoring_weights': self.scoring_weights,
            'supported_role_types': list(self.role_skill_mappings.keys()),
            'total_known_skills': sum(len(skills) for skills in self.role_skill_mappings.values())
        }
    
    def update_scoring_weights(self, new_weights: Dict):
        """Update scoring weights for different criteria"""
        # Validate weights sum to 1.0
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            normalized_weights = {k: v/total_weight for k, v in new_weights.items()}
            self.scoring_weights.update(normalized_weights)
        else:
            self.scoring_weights.update(new_weights)
        
        logger.info(f"Updated scoring weights: {self.scoring_weights}")

# Utility functions for testing and development
def test_ranking_engine():
    """Test function to verify ranking engine functionality"""
    engine = RankingEngine()
    info = engine.get_model_info()
    
    print("🤖 AI Ranking Engine Test")
    print("=" * 30)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n✅ Ranking engine initialized successfully!")
    return engine

if __name__ == "__main__":
    test_ranking_engine()