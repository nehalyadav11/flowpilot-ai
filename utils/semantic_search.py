"""
🔍 Semantic Search Engine
Advanced semantic search using FAISS vector database and transformer embeddings.

Features:
- Vector-based similarity search
- FAISS indexing for fast retrieval
- Multi-modal search (skills, experience, roles)
- Real-time candidate matching
- Similarity scoring and ranking
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import pickle
import os
from datetime import datetime

# Core libraries
try:
    import faiss
except ImportError:
    print("FAISS not available. Install: pip install faiss-cpu")
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("ML libraries not available. Install: pip install sentence-transformers scikit-learn")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSearch:
    """
    Semantic search engine for intelligent candidate matching
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the semantic search engine"""
        self.model_name = model_name
        self.sentence_model = None
        self.index = None
        self.candidate_embeddings = None
        self.candidate_metadata = []
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        
        # Search configuration
        self.similarity_threshold = 0.6
        self.max_results = 10
        
        # Index file path for persistence
        self.index_file = "data/search_index.faiss"
        self.metadata_file = "data/candidate_metadata.pkl"
        
        self._initialize_model()
        self._load_existing_index()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            self.sentence_model = SentenceTransformer(self.model_name)
            self.dimension = self.sentence_model.get_sentence_embedding_dimension()
            logger.info(f"✅ Sentence Transformer loaded: {self.model_name} (dim: {self.dimension})")
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer: {e}")
    
    def _load_existing_index(self):
        """Load existing FAISS index if available"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # Load FAISS index
                if faiss:
                    self.index = faiss.read_index(self.index_file)
                    logger.info(f"✅ Loaded existing FAISS index with {self.index.ntotal} vectors")
                
                # Load metadata
                with open(self.metadata_file, 'rb') as f:
                    self.candidate_metadata = pickle.load(f)
                    logger.info(f"✅ Loaded metadata for {len(self.candidate_metadata)} candidates")
            else:
                self._create_new_index()
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing index: {e}")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        try:
            if faiss and self.dimension:
                # Create a new index (using Inner Product for cosine similarity)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.candidate_metadata = []
                logger.info(f"✅ Created new FAISS index (dimension: {self.dimension})")
            else:
                logger.warning("⚠️ FAISS not available, using fallback search")
                
        except Exception as e:
            logger.error(f"❌ Failed to create FAISS index: {e}")
    
    def build_index(self, candidates: List[Dict]):
        """
        Build or update the search index with candidate data
        
        Args:
            candidates: List of candidate dictionaries
        """
        if not candidates:
            logger.warning("No candidates provided for indexing")
            return
        
        if not self.sentence_model:
            logger.error("Sentence transformer model not available")
            return
        
        try:
            logger.info(f"Building search index for {len(candidates)} candidates...")
            
            # Create candidate text representations
            candidate_texts = []
            valid_candidates = []
            
            for candidate in candidates:
                text_repr = self._create_candidate_search_text(candidate)
                if text_repr:
                    candidate_texts.append(text_repr)
                    valid_candidates.append(candidate)
            
            if not candidate_texts:
                logger.warning("No valid candidate texts for indexing")
                return
            
            # Generate embeddings
            embeddings = self.sentence_model.encode(
                candidate_texts,
                show_progress_bar=True,
                batch_size=32
            )
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Update or create index
            if faiss:
                self._update_faiss_index(embeddings, valid_candidates)
            else:
                self._update_fallback_index(embeddings, valid_candidates)
            
            logger.info(f"✅ Successfully indexed {len(valid_candidates)} candidates")
            
        except Exception as e:
            logger.error(f"❌ Error building search index: {e}")
    
    def _create_candidate_search_text(self, candidate: Dict) -> str:
        """Create searchable text representation of candidate"""
        text_parts = []
        
        # Add name (important for direct name searches)
        name = candidate.get('name', '').strip()
        if name and name.lower() != 'unknown':
            text_parts.append(f"Name: {name}")
        
        # Add skills (most important for search)
        skills = candidate.get('skills', [])
        if skills:
            text_parts.append(f"Skills: {' '.join(skills)}")
        
        # Add technical skills specifically
        technical_skills = candidate.get('technical_skills', [])
        if technical_skills:
            text_parts.append(f"Technical: {' '.join(technical_skills)}")
        
        # Add experience text
        experience = candidate.get('experience', '').strip()
        if experience:
            # Limit experience text to avoid overwhelming the embedding
            experience_short = experience[:200]
            text_parts.append(f"Experience: {experience_short}")
        
        # Add recent roles
        roles = candidate.get('recent_roles', [])
        if roles:
            text_parts.append(f"Roles: {' '.join(roles)}")
        
        # Add education
        education = candidate.get('education', '').strip()
        if education:
            education_short = education[:100]
            text_parts.append(f"Education: {education_short}")
        
        # Add degrees
        degrees = candidate.get('degrees', [])
        if degrees:
            text_parts.append(f"Degrees: {' '.join(degrees)}")
        
        # Add location if available
        location = candidate.get('location', '').strip()
        if location:
            text_parts.append(f"Location: {location}")
        
        return ' '.join(text_parts)
    
    def _update_faiss_index(self, embeddings: np.ndarray, candidates: List[Dict]):
        """Update FAISS index with new embeddings"""
        try:
            if self.index is None:
                self._create_new_index()
            
            # Add embeddings to index
            self.index.add(embeddings.astype(np.float32))
            
            # Update metadata
            for i, candidate in enumerate(candidates):
                metadata = {
                    'index_id': len(self.candidate_metadata),
                    'name': candidate.get('name', 'Unknown'),
                    'email': candidate.get('email', ''),
                    'phone': candidate.get('phone', ''),
                    'skills': candidate.get('skills', []),
                    'experience': candidate.get('total_years', 0),
                    'filename': candidate.get('filename', ''),
                    'completeness_score': candidate.get('completeness_score', 0),
                    'indexed_date': datetime.now().isoformat(),
                    'search_text': self._create_candidate_search_text(candidate)
                }
                self.candidate_metadata.append(metadata)
            
            # Save index and metadata
            self._save_index()
            
        except Exception as e:
            logger.error(f"Error updating FAISS index: {e}")
    
    def _update_fallback_index(self, embeddings: np.ndarray, candidates: List[Dict]):
        """Update fallback index (when FAISS not available)"""
        self.candidate_embeddings = embeddings
        self.candidate_metadata = []
        
        for i, candidate in enumerate(candidates):
            metadata = {
                'index_id': i,
                'name': candidate.get('name', 'Unknown'),
                'email': candidate.get('email', ''),
                'phone': candidate.get('phone', ''),
                'skills': candidate.get('skills', []),
                'experience': candidate.get('total_years', 0),
                'filename': candidate.get('filename', ''),
                'completeness_score': candidate.get('completeness_score', 0),
                'indexed_date': datetime.now().isoformat(),
                'search_text': self._create_candidate_search_text(candidate)
            }
            self.candidate_metadata.append(metadata)
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            
            # Save FAISS index
            if faiss and self.index:
                faiss.write_index(self.index, self.index_file)
            
            # Save metadata
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.candidate_metadata, f)
            
            logger.info("✅ Search index and metadata saved to disk")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not save index to disk: {e}")
    
    def search(self, query: str, candidates: Optional[List[Dict]] = None, limit: int = None) -> List[Dict]:
        """
        Perform semantic search for candidates matching the query
        
        Args:
            query: Search query string
            candidates: Optional list of candidates to search in (for real-time search)
            limit: Maximum number of results to return
            
        Returns:
            List[Dict]: Ranked search results with similarity scores
        """
        if not query or len(query.strip()) < 2:
            return []
        
        if not self.sentence_model:
            logger.error("Sentence transformer model not available for search")
            return []
        
        try:
            # If candidates provided, do real-time search
            if candidates:
                return self._realtime_search(query, candidates, limit)
            
            # Otherwise use indexed search
            return self._indexed_search(query, limit)
            
        except Exception as e:
            logger.error(f"❌ Error during search: {e}")
            return []
    
    def _realtime_search(self, query: str, candidates: List[Dict], limit: int = None) -> List[Dict]:
        """Perform real-time search on provided candidates"""
        if not candidates:
            return []
        
        limit = limit or self.max_results
        
        # Create candidate texts and embeddings
        candidate_texts = []
        valid_candidates = []
        
        for candidate in candidates:
            text_repr = self._create_candidate_search_text(candidate)
            if text_repr:
                candidate_texts.append(text_repr)
                valid_candidates.append(candidate)
        
        if not candidate_texts:
            return []
        
        # Generate embeddings for query and candidates
        query_embedding = self.sentence_model.encode([query])
        candidate_embeddings = self.sentence_model.encode(candidate_texts)
        
        # Normalize embeddings
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        candidate_embeddings = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Create results with similarity scores
        results = []
        for i, candidate in enumerate(valid_candidates):
            similarity = similarities[i]
            
            if similarity >= self.similarity_threshold:
                result = {
                    'name': candidate.get('name', 'Unknown'),
                    'email': candidate.get('email', ''),
                    'phone': candidate.get('phone', ''),
                    'skills': candidate.get('skills', []),
                    'experience': candidate.get('total_years', 0),
                    'filename': candidate.get('filename', ''),
                    'similarity': float(similarity),
                    'match_score': int(similarity * 100),
                    'search_date': datetime.now().isoformat()
                }
                results.append(result)
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
    def _indexed_search(self, query: str, limit: int = None) -> List[Dict]:
        """Perform search using pre-built index"""
        if not self.candidate_metadata:
            logger.warning("No candidates in search index")
            return []
        
        limit = limit or self.max_results
        
        # Generate query embedding
        query_embedding = self.sentence_model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        if faiss and self.index:
            return self._faiss_search(query_embedding, limit)
        else:
            return self._fallback_search(query_embedding, limit)
    
    def _faiss_search(self, query_embedding: np.ndarray, limit: int) -> List[Dict]:
        """Search using FAISS index"""
        try:
            # Search FAISS index
            similarities, indices = self.index.search(query_embedding.astype(np.float32), limit)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1 or similarity < self.similarity_threshold:  # Invalid result
                    continue
                
                if idx < len(self.candidate_metadata):
                    metadata = self.candidate_metadata[idx]
                    result = {
                        'name': metadata['name'],
                        'email': metadata['email'],
                        'phone': metadata['phone'],
                        'skills': metadata['skills'],
                        'experience': metadata['experience'],
                        'filename': metadata['filename'],
                        'similarity': float(similarity),
                        'match_score': int(similarity * 100),
                        'search_date': datetime.now().isoformat()
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")
            return []
    
    def _fallback_search(self, query_embedding: np.ndarray, limit: int) -> List[Dict]:
        """Fallback search when FAISS not available"""
        if self.candidate_embeddings is None:
            return []
        
        try:
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.candidate_embeddings)[0]
            
            # Create results
            results = []
            for i, similarity in enumerate(similarities):
                if similarity >= self.similarity_threshold and i < len(self.candidate_metadata):
                    metadata = self.candidate_metadata[i]
                    result = {
                        'name': metadata['name'],
                        'email': metadata['email'],
                        'phone': metadata['phone'],
                        'skills': metadata['skills'],
                        'experience': metadata['experience'],
                        'filename': metadata['filename'],
                        'similarity': float(similarity),
                        'match_score': int(similarity * 100),
                        'search_date': datetime.now().isoformat()
                    }
                    results.append(result)
            
            # Sort by similarity and limit
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return []
    
    def search_by_skills(self, skills: List[str], limit: int = None) -> List[Dict]:
        """
        Search for candidates with specific skills
        
        Args:
            skills: List of skill names to search for
            limit: Maximum number of results
            
        Returns:
            List[Dict]: Matching candidates
        """
        if not skills:
            return []
        
        # Create skill-focused query
        query = f"Skills: {' '.join(skills)}"
        return self.search(query, limit=limit)
    
    def search_by_role(self, role: str, limit: int = None) -> List[Dict]:
        """
        Search for candidates by role/job title
        
        Args:
            role: Job role or title to search for
            limit: Maximum number of results
            
        Returns:
            List[Dict]: Matching candidates
        """
        if not role:
            return []
        
        # Create role-focused query
        query = f"Role: {role} Experience: {role}"
        return self.search(query, limit=limit)
    
    def search_similar_candidates(self, reference_candidate: Dict, limit: int = None) -> List[Dict]:
        """
        Find candidates similar to a reference candidate
        
        Args:
            reference_candidate: Candidate to use as reference
            limit: Maximum number of results
            
        Returns:
            List[Dict]: Similar candidates
        """
        # Create query from reference candidate
        query = self._create_candidate_search_text(reference_candidate)
        
        if not query:
            return []
        
        results = self.search(query, limit=limit)
        
        # Filter out the reference candidate itself
        reference_name = reference_candidate.get('name', '').lower()
        if reference_name:
            results = [r for r in results if r['name'].lower() != reference_name]
        
        return results
    
    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """
        Get search suggestions based on partial query
        
        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions
            
        Returns:
            List[str]: Search suggestions
        """
        if not partial_query or len(partial_query) < 2:
            return []
        
        suggestions = set()
        partial_lower = partial_query.lower()
        
        # Extract suggestions from candidate metadata
        for metadata in self.candidate_metadata:
            # Check skills
            for skill in metadata.get('skills', []):
                if partial_lower in skill.lower():
                    suggestions.add(skill)
            
            # Check name
            name = metadata.get('name', '')
            if partial_lower in name.lower():
                suggestions.add(name)
        
        # Add common search suggestions
        common_suggestions = [
            'python developer', 'data scientist', 'machine learning', 
            'web developer', 'software engineer', 'react', 'java',
            'javascript', 'sql', 'aws', 'docker', 'kubernetes'
        ]
        
        for suggestion in common_suggestions:
            if partial_lower in suggestion.lower():
                suggestions.add(suggestion)
        
        return sorted(list(suggestions))[:limit]
    
    def get_statistics(self) -> Dict:
        """Get search engine statistics"""
        return {
            'model_name': self.model_name,
            'model_loaded': bool(self.sentence_model),
            'faiss_available': bool(faiss),
            'index_size': self.index.ntotal if (faiss and self.index) else len(self.candidate_metadata),
            'total_candidates': len(self.candidate_metadata),
            'embedding_dimension': self.dimension,
            'similarity_threshold': self.similarity_threshold,
            'max_results': self.max_results,
            'index_file_exists': os.path.exists(self.index_file),
            'metadata_file_exists': os.path.exists(self.metadata_file)
        }
    
    def update_settings(self, similarity_threshold: float = None, max_results: int = None):
        """Update search settings"""
        if similarity_threshold is not None:
            self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))
            logger.info(f"Updated similarity threshold to {self.similarity_threshold}")
        
        if max_results is not None:
            self.max_results = max(1, max_results)
            logger.info(f"Updated max results to {self.max_results}")
    
    def clear_index(self):
        """Clear the search index and metadata"""
        try:
            if faiss and self.index:
                self.index.reset()
            
            self.candidate_metadata = []
            self.candidate_embeddings = None
            
            # Remove index files
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
            
            logger.info("✅ Search index cleared")
            
        except Exception as e:
            logger.error(f"Error clearing search index: {e}")

# Utility functions for testing and development
def test_semantic_search():
    """Test function to verify semantic search functionality"""
    search_engine = SemanticSearch()
    stats = search_engine.get_statistics()
    
    print("🔍 Semantic Search Engine Test")
    print("=" * 30)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Semantic search engine initialized successfully!")
    return search_engine

if __name__ == "__main__":
    test_semantic_search()