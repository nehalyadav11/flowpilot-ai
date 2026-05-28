"""
📄 Resume Parser Module
Advanced resume parsing with multi-format support, NER extraction, and language detection.

Features:
- PDF, TXT, DOCX parsing
- Named Entity Recognition for skills, emails, phones
- Multi-language detection and translation
- Experience and education extraction
- Semantic skill matching
"""

import re
import io
import os
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

# Core libraries
import pandas as pd
import numpy as np

# File processing
try:
    import PyPDF2
    import pdfminer.high_level as pdfminer
    from pdfminer.high_level import extract_text as extract_pdf_text
except ImportError:
    print("PDF processing libraries not available. Install: pip install PyPDF2 pdfminer.six")

try:
    import docx
except ImportError:
    print("DOCX processing not available. Install: pip install python-docx")

# NLP and ML
try:
    import spacy
    from spacy import displacy
except ImportError:
    print("spaCy not available. Install: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
except ImportError:
    print("Transformers not available. Install: pip install transformers torch")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Sentence Transformers not available. Install: pip install sentence-transformers")

# Language detection and translation
try:
    from langdetect import detect, detect_langs
except ImportError:
    print("Language detection not available. Install: pip install langdetect")

try:
    from googletrans import Translator
except ImportError:
    print("Google Translate not available. Install: pip install googletrans==4.0.0rc1")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeParser:
    """
    Advanced resume parser with AI-powered extraction and multi-language support
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the resume parser with NLP models"""
        self.model_name = model_name
        self.translator = None
        self.nlp = None
        self.sentence_model = None
        self.ner_pipeline = None
        
        # Skill categories for better classification
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'kotlin'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'matplotlib'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'tools': ['git', 'jira', 'confluence', 'slack', 'trello', 'figma', 'photoshop']
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models and tools"""
        try:
            # Initialize Google Translator
            self.translator = Translator()
            logger.info("✅ Google Translator initialized")
        except Exception as e:
            logger.warning(f"⚠️ Translator initialization failed: {e}")
        
        try:
            # Initialize spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded")
        except Exception as e:
            logger.warning(f"⚠️ spaCy model loading failed: {e}")
            try:
                # Try to download and load
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model downloaded and loaded")
            except:
                logger.error("❌ Could not load spaCy model")
        
        try:
            # Initialize Sentence Transformers
            self.sentence_model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Sentence Transformer {self.model_name} loaded")
        except Exception as e:
            logger.warning(f"⚠️ Sentence Transformer loading failed: {e}")
        
        try:
            # Initialize NER pipeline for better entity recognition
            self.ner_pipeline = pipeline("ner", 
                                       model="dbmdz/bert-base-cased-finetuned-conll03-english",
                                       aggregation_strategy="simple")
            logger.info("✅ NER pipeline initialized")
        except Exception as e:
            logger.warning(f"⚠️ NER pipeline initialization failed: {e}")
    
    def parse_file(self, file) -> Dict:
        """
        Parse a resume file and extract structured information
        
        Args:
            file: Streamlit uploaded file object
            
        Returns:
            Dict: Structured resume data
        """
        try:
            # Extract text based on file type
            if file.type == "application/pdf":
                text = self._extract_pdf_text(file)
            elif file.type == "text/plain":
                text = self._extract_txt_text(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = self._extract_docx_text(file)
            else:
                raise ValueError(f"Unsupported file type: {file.type}")
            
            if not text or len(text.strip()) < 50:
                raise ValueError("Extracted text is too short or empty")
            
            # Detect language and translate if needed
            detected_language, translated_text = self._detect_and_translate(text)
            
            # Extract structured information
            candidate_data = {
                'filename': file.name,
                'original_language': detected_language,
                'raw_text': text,
                'processed_text': translated_text,
                'extraction_date': datetime.now().isoformat(),
                **self._extract_contact_info(translated_text),
                **self._extract_skills(translated_text),
                **self._extract_experience(translated_text),
                **self._extract_education(translated_text),
                **self._extract_additional_info(translated_text)
            }
            
            # Calculate completeness score
            candidate_data['completeness_score'] = self._calculate_completeness(candidate_data)
            
            logger.info(f"✅ Successfully parsed {file.name}")
            return candidate_data
            
        except Exception as e:
            logger.error(f"❌ Error parsing {file.name}: {str(e)}")
            return {
                'filename': file.name,
                'error': str(e),
                'extraction_date': datetime.now().isoformat(),
                'name': 'Unknown',
                'email': '',
                'phone': '',
                'skills': [],
                'experience': '',
                'education': '',
                'completeness_score': 0
            }
    
    def _extract_pdf_text(self, file) -> str:
        """Extract text from PDF file"""
        try:
            # Try pdfminer first (more reliable)
            file.seek(0)
            text = extract_pdf_text(file)
            return text
        except Exception as e:
            logger.warning(f"pdfminer failed, trying PyPDF2: {e}")
            try:
                # Fallback to PyPDF2
                file.seek(0)
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed: {e2}")
                raise ValueError("Could not extract text from PDF")
    
    def _extract_txt_text(self, file) -> str:
        """Extract text from TXT file"""
        try:
            file.seek(0)
            text = file.read().decode('utf-8', errors='ignore')
            return text
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            raise ValueError("Could not extract text from TXT file")
    
    def _extract_docx_text(self, file) -> str:
        """Extract text from DOCX file"""
        try:
            file.seek(0)
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise ValueError("Could not extract text from DOCX file")
    
    def _detect_and_translate(self, text: str) -> tuple:
        """Detect language and translate to English if needed"""
        try:
            # Detect language
            detected_lang = detect(text)
            logger.info(f"Detected language: {detected_lang}")
            
            if detected_lang == 'en':
                return detected_lang, text
            
            # Translate to English
            if self.translator:
                # Split text into chunks to avoid API limits
                chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
                translated_chunks = []
                
                for chunk in chunks:
                    try:
                        translation = self.translator.translate(chunk, src=detected_lang, dest='en')
                        translated_chunks.append(translation.text)
                    except Exception as e:
                        logger.warning(f"Translation chunk failed: {e}")
                        translated_chunks.append(chunk)  # Use original if translation fails
                
                translated_text = " ".join(translated_chunks)
                logger.info(f"✅ Translated from {detected_lang} to English")
                return detected_lang, translated_text
            else:
                logger.warning("Translator not available, using original text")
                return detected_lang, text
                
        except Exception as e:
            logger.warning(f"Language detection/translation failed: {e}")
            return 'unknown', text
    
    def _extract_contact_info(self, text: str) -> Dict:
        """Extract contact information using regex and NER"""
        contact_info = {
            'name': '',
            'email': '',
            'phone': '',
            'location': '',
            'linkedin': '',
            'github': ''
        }
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Extract phone number
        phone_patterns = [
            r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            r'\+?[0-9]{1,4}?[-.\s]?\(?[0-9]{1,3}\)?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}'
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                contact_info['phone'] = phones[0].strip()
                break
        
        # Extract LinkedIn
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_matches = re.findall(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_matches:
            contact_info['linkedin'] = linkedin_matches[0]
        
        # Extract GitHub
        github_pattern = r'github\.com/[\w-]+'
        github_matches = re.findall(github_pattern, text, re.IGNORECASE)
        if github_matches:
            contact_info['github'] = github_matches[0]
        
        # Extract name using NER or fallback methods
        contact_info['name'] = self._extract_name(text)
        
        # Extract location
        contact_info['location'] = self._extract_location(text)
        
        return contact_info
    
    def _extract_name(self, text: str) -> str:
        """Extract candidate name using multiple strategies"""
        try:
            # Strategy 1: Use NER if available
            if self.ner_pipeline:
                entities = self.ner_pipeline(text[:1000])  # Check first 1000 chars
                for entity in entities:
                    if entity['entity_group'] == 'PER' and entity['score'] > 0.9:
                        return entity['word'].strip()
            
            # Strategy 2: Use spaCy NER
            if self.nlp:
                doc = self.nlp(text[:1000])
                for ent in doc.ents:
                    if ent.label_ == "PERSON" and len(ent.text.split()) <= 3:
                        return ent.text.strip()
            
            # Strategy 3: Look for patterns at the beginning
            lines = text.split('\n')[:10]  # Check first 10 lines
            for line in lines:
                line = line.strip()
                # Skip common headers
                if any(header in line.lower() for header in ['resume', 'curriculum', 'cv', 'contact', 'email']):
                    continue
                
                # Look for name patterns
                words = line.split()
                if 2 <= len(words) <= 3 and all(word.isalpha() and word[0].isupper() for word in words):
                    return line
            
            # Strategy 4: Fallback - use first line that looks like a name
            for line in text.split('\n')[:5]:
                line = line.strip()
                if len(line) > 3 and len(line) < 50 and not '@' in line and not '.' in line:
                    words = line.split()
                    if 1 <= len(words) <= 3:
                        return line
            
            return "Unknown"
            
        except Exception as e:
            logger.warning(f"Name extraction failed: {e}")
            return "Unknown"
    
    def _extract_location(self, text: str) -> str:
        """Extract location information"""
        try:
            # Common location patterns
            location_patterns = [
                r'([A-Z][a-z]+,\s*[A-Z]{2})',  # City, State
                r'([A-Z][a-z\s]+,\s*[A-Z][a-z\s]+)',  # City, Country
            ]
            
            for pattern in location_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    return matches[0]
            
            # Use NER to find locations
            if self.nlp:
                doc = self.nlp(text[:1000])
                for ent in doc.ents:
                    if ent.label_ in ["GPE", "LOC"]:  # Geopolitical entities or locations
                        return ent.text.strip()
            
            return ""
            
        except Exception as e:
            logger.warning(f"Location extraction failed: {e}")
            return ""
    
    def _extract_skills(self, text: str) -> Dict:
        """Extract skills and categorize them"""
        skills_info = {
            'skills': [],
            'technical_skills': [],
            'soft_skills': [],
            'skill_categories': {}
        }
        
        try:
            text_lower = text.lower()
            
            # Extract technical skills
            all_skills = []
            for category, skill_list in self.skill_categories.items():
                found_skills = []
                for skill in skill_list:
                    # Look for exact matches and variations
                    patterns = [
                        rf'\b{re.escape(skill)}\b',
                        rf'\b{re.escape(skill)}\.js\b',
                        rf'\b{re.escape(skill)}\.py\b',
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, text_lower):
                            found_skills.append(skill)
                            all_skills.append(skill)
                            break
                
                if found_skills:
                    skills_info['skill_categories'][category] = found_skills
            
            # Additional skill extraction using regex patterns
            additional_skills = self._extract_additional_skills(text)
            all_skills.extend(additional_skills)
            
            # Remove duplicates and sort
            skills_info['skills'] = sorted(list(set(all_skills)))
            skills_info['technical_skills'] = [s for s in skills_info['skills'] 
                                             if any(s in cat for cat in self.skill_categories.values())]
            
            # Extract soft skills (basic implementation)
            soft_skills = ['leadership', 'communication', 'teamwork', 'problem solving', 
                          'analytical', 'creative', 'adaptable', 'organized']
            found_soft_skills = []
            for skill in soft_skills:
                if skill in text_lower:
                    found_soft_skills.append(skill)
            
            skills_info['soft_skills'] = found_soft_skills
            
            return skills_info
            
        except Exception as e:
            logger.error(f"Skills extraction failed: {e}")
            return skills_info
    
    def _extract_additional_skills(self, text: str) -> List[str]:
        """Extract additional skills using patterns"""
        additional_skills = []
        
        # Common skill indicators
        skill_sections = re.findall(r'(?:skills?|technologies?|tools?)[:\-\s]*(.*?)(?:\n\n|\n[A-Z])', 
                                   text, re.IGNORECASE | re.DOTALL)
        
        for section in skill_sections:
            # Split by common delimiters
            skills = re.split(r'[,;•\n\t]+', section.strip())
            for skill in skills:
                skill = skill.strip().lower()
                if skill and 2 <= len(skill) <= 30 and not skill.isdigit():
                    additional_skills.append(skill)
        
        return additional_skills[:20]  # Limit to top 20
    
    def _extract_experience(self, text: str) -> Dict:
        """Extract work experience information"""
        experience_info = {
            'experience': '',
            'total_years': 0,
            'recent_roles': [],
            'companies': []
        }
        
        try:
            # Find experience sections
            exp_patterns = [
                r'(?:experience|employment|work history)[:\-\s]*(.*?)(?=\n(?:education|skills|projects|certifications)|\Z)',
                r'(?:professional experience)[:\-\s]*(.*?)(?=\n(?:education|skills|projects|certifications)|\Z)'
            ]
            
            experience_text = ""
            for pattern in exp_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                if matches:
                    experience_text = matches[0].strip()
                    break
            
            if not experience_text:
                # Fallback: look for job titles and companies
                experience_text = self._find_experience_by_patterns(text)
            
            experience_info['experience'] = experience_text[:500]  # Limit length
            
            # Extract years of experience
            years_matches = re.findall(r'(\d+)\+?\s*years?', text, re.IGNORECASE)
            if years_matches:
                experience_info['total_years'] = max(int(year) for year in years_matches)
            
            # Extract recent roles (simplified)
            role_patterns = [
                r'([A-Z][a-z\s]+(?:engineer|developer|analyst|manager|director|specialist))',
                r'(software\s+[a-z\s]+)',
                r'(data\s+[a-z\s]+)'
            ]
            
            roles = []
            for pattern in role_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                roles.extend(matches[:3])  # Limit to 3 per pattern
            
            experience_info['recent_roles'] = list(set(roles))[:5]  # Max 5 roles
            
            return experience_info
            
        except Exception as e:
            logger.error(f"Experience extraction failed: {e}")
            return experience_info
    
    def _find_experience_by_patterns(self, text: str) -> str:
        """Find experience information using common patterns"""
        # Look for job titles followed by company names
        job_patterns = [
            r'((?:senior|junior|lead)?\s*(?:software|data|web|mobile|full[\s\-]?stack)?\s*(?:engineer|developer|analyst|scientist))[^\n]*',
            r'(manager|director|consultant|specialist)[^\n]*'
        ]
        
        experience_lines = []
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            experience_lines.extend(matches[:3])
        
        return '. '.join(experience_lines)
    
    def _extract_education(self, text: str) -> Dict:
        """Extract education information"""
        education_info = {
            'education': '',
            'degrees': [],
            'institutions': [],
            'graduation_year': None
        }
        
        try:
            # Find education section
            edu_patterns = [
                r'(?:education|academic|qualifications)[:\-\s]*(.*?)(?=\n(?:experience|skills|projects|certifications)|\Z)',
            ]
            
            education_text = ""
            for pattern in edu_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                if matches:
                    education_text = matches[0].strip()
                    break
            
            education_info['education'] = education_text[:300]  # Limit length
            
            # Extract degrees
            degree_patterns = [
                r'\b(bachelor|master|phd|doctorate|mba|bs|ms|ba|ma|btech|mtech)\b',
                r'\b(b\.?\s*[a-z]\.?|m\.?\s*[a-z]\.?)\b'
            ]
            
            degrees = []
            for pattern in degree_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                degrees.extend(matches)
            
            education_info['degrees'] = list(set(degrees))[:3]
            
            # Extract graduation year
            year_matches = re.findall(r'\b(19|20)\d{2}\b', education_text)
            if year_matches:
                years = [int(f"{match[0]}{match[1:]}" if isinstance(match, str) else f"{match[0]}{match[1:]}") for match in year_matches]
                education_info['graduation_year'] = max(years) if years else None
            
            return education_info
            
        except Exception as e:
            logger.error(f"Education extraction failed: {e}")
            return education_info
    
    def _extract_additional_info(self, text: str) -> Dict:
        """Extract additional information like certifications, projects, etc."""
        additional_info = {
            'certifications': [],
            'projects': [],
            'languages': [],
            'achievements': []
        }
        
        try:
            # Extract certifications
            cert_keywords = ['certification', 'certified', 'certificate', 'credential']
            for keyword in cert_keywords:
                pattern = rf'{keyword}[:\-\s]*([^\n]+)'
                matches = re.findall(pattern, text, re.IGNORECASE)
                additional_info['certifications'].extend(matches[:3])
            
            # Extract projects
            project_pattern = r'(?:projects?|portfolio)[:\-\s]*(.*?)(?=\n(?:[A-Z][a-z]+:|$))'
            project_matches = re.findall(project_pattern, text, re.IGNORECASE | re.DOTALL)
            if project_matches:
                projects = project_matches[0].split('\n')[:3]
                additional_info['projects'] = [p.strip() for p in projects if p.strip()]
            
            # Extract languages
            lang_pattern = r'(?:languages?)[:\-\s]*([^\n]+)'
            lang_matches = re.findall(lang_pattern, text, re.IGNORECASE)
            if lang_matches:
                languages = re.split(r'[,;]+', lang_matches[0])
                additional_info['languages'] = [lang.strip() for lang in languages if lang.strip()][:5]
            
            return additional_info
            
        except Exception as e:
            logger.error(f"Additional info extraction failed: {e}")
            return additional_info
    
    def _calculate_completeness(self, candidate_data: Dict) -> int:
        """Calculate completeness score based on available information"""
        score = 0
        max_score = 100
        
        # Contact information (30 points)
        if candidate_data.get('name', '').lower() != 'unknown':
            score += 10
        if candidate_data.get('email'):
            score += 10
        if candidate_data.get('phone'):
            score += 10
        
        # Skills (25 points)
        skills = candidate_data.get('skills', [])
        if len(skills) >= 5:
            score += 25
        elif len(skills) >= 3:
            score += 15
        elif len(skills) >= 1:
            score += 10
        
        # Experience (25 points)
        if candidate_data.get('experience'):
            score += 15
            if candidate_data.get('total_years', 0) > 0:
                score += 10
        
        # Education (20 points)
        if candidate_data.get('education'):
            score += 10
            if candidate_data.get('degrees'):
                score += 10
        
        return min(score, max_score)

    def get_skill_embeddings(self, skills: List[str]) -> np.ndarray:
        """Generate embeddings for skills using sentence transformer"""
        try:
            if self.sentence_model and skills:
                embeddings = self.sentence_model.encode(skills)
                return embeddings
            return np.array([])
        except Exception as e:
            logger.error(f"Skill embedding generation failed: {e}")
            return np.array([])
    
    def get_stats(self) -> Dict:
        """Get parser statistics and model information"""
        return {
            'model_name': self.model_name,
            'nlp_model': 'en_core_web_sm' if self.nlp else 'Not loaded',
            'translator_available': bool(self.translator),
            'sentence_transformer_available': bool(self.sentence_model),
            'ner_pipeline_available': bool(self.ner_pipeline),
            'supported_formats': ['PDF', 'TXT', 'DOCX'],
            'skill_categories': len(self.skill_categories),
            'total_predefined_skills': sum(len(skills) for skills in self.skill_categories.values())
        }

# Utility functions for testing and development
def test_parser():
    """Test function to verify parser functionality"""
    parser = ResumeParser()
    stats = parser.get_stats()
    
    print("🚀 Resume Parser Test")
    print("=" * 30)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Parser initialized successfully!")
    return parser

if __name__ == "__main__":
    test_parser()