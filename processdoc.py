import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re
from typing import List, Dict, Tuple
from collections import Counter
import fitz  # PyMuPDF

class AdaptiveTroubleshootingDocumentProcessor:
    def __init__(self, model_name="all-mpnet-base-v2"):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)
        self.chunks = []
        self.metadata = []
        self.document_patterns = {}
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with better formatting preservation"""
        doc = fitz.open(file_path)
        text = ""
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            # Clean up common PDF artifacts
            page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
            page_text = re.sub(r'(\w)-\s*\n(\w)', r'\1\2', page_text)  # Fix hyphenated words
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        doc.close()
        return text
    
    def analyze_document_structure(self, text: str) -> Dict:
        """Analyze the document to find natural splitting patterns"""
        patterns = {
            'numbered_lists': [],
            'bullet_points': [],
            'headers': [],
            'sections': [],
            'steps': [],
            'custom_markers': []
        }
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            # Find numbered items (1., 2., etc.)
            if re.match(r'^\d+[\.\)]\s+', line):
                patterns['numbered_lists'].append(line[:50])
            
            # Find lettered items (a., b., etc.)
            elif re.match(r'^[a-zA-Z][\.\)]\s+', line):
                patterns['numbered_lists'].append(line[:50])
            
            # Find bullet points
            elif re.match(r'^[•·▪▫◦‣⁃\-\*]\s+', line):
                patterns['bullet_points'].append(line[:50])
            
            # Find potential headers (short lines, mostly caps or title case)
            elif len(line) < 100 and (line.isupper() or line.istitle()) and not line.endswith('.'):
                patterns['headers'].append(line)
            
            # Find step indicators
            elif re.match(r'^(step|stage|phase)\s*\d+', line, re.IGNORECASE):
                patterns['steps'].append(line[:50])
            
            # Find section markers
            elif re.match(r'^(section|chapter|part)\s*\d+', line, re.IGNORECASE):
                patterns['sections'].append(line[:50])
            
            # Find other potential markers (lines that start with capital words followed by colon)
            elif re.match(r'^[A-Z][A-Za-z\s]{2,15}:\s*', line):
                patterns['custom_markers'].append(line[:50])
        
        # Count occurrences to find most common patterns
        for key in patterns:
            if patterns[key]:
                counter = Counter(patterns[key])
                patterns[key] = counter.most_common(10)
        
        return patterns
    
    def print_document_analysis(self, patterns: Dict):
        """Print analysis of document structure"""
        print("\n=== DOCUMENT STRUCTURE ANALYSIS ===")
        
        for pattern_type, items in patterns.items():
            if items:
                print(f"\n{pattern_type.upper().replace('_', ' ')}:")
                for item, count in items:
                    print(f"  • {item} (appears {count} times)")
        
        print("\n" + "="*50)
    
    def create_adaptive_splitting_patterns(self, patterns: Dict) -> List[str]:
        """Create regex patterns based on document analysis"""
        regex_patterns = []
        
        # Always include basic patterns
        regex_patterns.extend([
            r'\n(?=#{1,6}\s)',  # Markdown headers
            r'\n(?=\d+\.\s+[A-Z])',  # Numbered sections starting with capital
        ])
        
        # Add patterns based on what we found
        if patterns['numbered_lists']:
            regex_patterns.append(r'\n(?=\d+[\.\)]\s+)')  # Numbered lists
            regex_patterns.append(r'\n(?=[a-zA-Z][\.\)]\s+)')  # Lettered lists
        
        if patterns['bullet_points']:
            regex_patterns.append(r'\n(?=[•·▪▫◦‣⁃\-\*]\s+)')  # Bullet points
        
        if patterns['steps']:
            regex_patterns.append(r'\n(?=(?:step|stage|phase)\s*\d+)', re.IGNORECASE)
        
        if patterns['sections']:
            regex_patterns.append(r'\n(?=(?:section|chapter|part)\s*\d+)', re.IGNORECASE)
        
        
        return regex_patterns
    
    def sliding_window_chunking(self, text: str, source: str, chunk_size: int = 600, overlap: int = 100) -> List[Tuple[str, Dict]]:
        """Fallback chunking method using sliding window with sentence awareness"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        sentence_start = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) > chunk_size and current_chunk:
                # Save current chunk
                metadata = {
                    'source': source,
                    'chunk_type': 'sliding_window',
                    'sentence_start': sentence_start,
                    'sentence_end': i - 1,
                    'length': len(current_chunk),
                    'word_count': len(current_chunk.split())
                }
                chunks.append((current_chunk.strip(), metadata))
                
                # Start new chunk with overlap
                if overlap > 0:
                    words = current_chunk.split()
                    overlap_words = words[-overlap:] if len(words) > overlap else words
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    current_chunk = sentence
                sentence_start = max(0, i - overlap // 10)  # Approximate sentence overlap
            else:
                current_chunk = test_chunk
        
        # Don't forget the last chunk
        if current_chunk.strip():
            metadata = {
                'source': source,
                'chunk_type': 'sliding_window',
                'sentence_start': sentence_start,
                'sentence_end': len(sentences) - 1,
                'length': len(current_chunk),
                'word_count': len(current_chunk.split())
            }
            chunks.append((current_chunk.strip(), metadata))
        
        return chunks
    
    def adaptive_chunk_document(self, text: str, source: str, chunk_size: int = 600, overlap: int = 100) -> List[Tuple[str, Dict]]:
        """Adaptive chunking based on document structure analysis"""
        
        # First, analyze the document
        patterns = self.analyze_document_structure(text)
        self.document_patterns[source] = patterns
        
        # Print analysis for user review
        self.print_document_analysis(patterns)
        
        # Create splitting patterns
        splitting_patterns = self.create_adaptive_splitting_patterns(patterns)
        
        if not splitting_patterns:
            print(f"No structural patterns found in {source}. Using sliding window chunking.")
            return self.sliding_window_chunking(text, source, chunk_size, overlap)
        
        print(f"Using {len(splitting_patterns)} splitting patterns for {source}")
        
        # Apply splitting patterns
        sections = [text.strip()]
        
        for pattern in splitting_patterns:
            new_sections = []
            for section in sections:
                try:
                    split_sections = re.split(pattern, section, flags=re.IGNORECASE if 'IGNORECASE' in str(pattern) else 0)
                    new_sections.extend([s.strip() for s in split_sections if s.strip()])
                except:
                    new_sections.append(section)  # If pattern fails, keep original
            sections = new_sections
        
        # Remove very short sections
        sections = [s for s in sections if len(s) > 50]
        
        print(f"Document split into {len(sections)} sections")
        
        # Process sections into chunks
        chunks = []
        for section_idx, section in enumerate(sections):
            # Extract potential title (first line if it's short)
            lines = section.split('\n', 1)
            title = lines[0] if len(lines) > 1 and len(lines[0]) < 100 else ""
            
            if len(section) <= chunk_size:
                # Section fits in one chunk
                metadata = {
                    'source': source,
                    'chunk_type': 'complete_section',
                    'section_index': section_idx,
                    'title': title,
                    'length': len(section),
                    'word_count': len(section.split())
                }
                chunks.append((section, metadata))
            else:
                # Use sliding window for large sections
                section_chunks = self.sliding_window_chunking(section, source, chunk_size, overlap)
                # Update metadata to include section info
                for chunk_text, chunk_metadata in section_chunks:
                    chunk_metadata.update({
                        'parent_section': section_idx,
                        'section_title': title,
                        'chunk_type': 'partial_section'
                    })
                    chunks.append((chunk_text, chunk_metadata))
        
        return chunks
    
    def process_pdf_file(self, file_path: str) -> List[Tuple[str, Dict]]:
        """Process a PDF file with adaptive chunking"""
        content = self.extract_text_from_pdf(file_path)
        source_name = Path(file_path).stem
        return self.adaptive_chunk_document(content, source_name)
    
    def process_directory(self, directory_path: str) -> None:
        """Process all PDF files in directory"""
        directory = Path(directory_path)
        if not directory.exists():
            print(f"Directory {directory_path} does not exist!")
            return
        
        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            print(f"No .pdf files found in {directory_path}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        all_chunks = []
        
        for file_path in pdf_files:
            print(f"\n{'='*60}")
            print(f"Processing: {file_path.name}")
            print('='*60)
            
            file_chunks = self.process_pdf_file(str(file_path))
            all_chunks.extend(file_chunks)
            print(f"→ Generated {len(file_chunks)} chunks from {file_path.name}")
        
        self.chunks = [chunk for chunk, _ in all_chunks]
        self.metadata = [metadata for _, metadata in all_chunks]
        
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print('='*60)
        print(f"Total chunks created: {len(self.chunks)}")
        print(f"Average chunk length: {np.mean([len(chunk) for chunk in self.chunks]):.0f} characters")
        print(f"Average word count: {np.mean([meta.get('word_count', 0) for meta in self.metadata]):.0f} words")
        
        # Show chunk type distribution
        chunk_types = Counter([meta.get('chunk_type', 'unknown') for meta in self.metadata])
        print(f"\nChunk type distribution:")
        for chunk_type, count in chunk_types.items():
            print(f"  • {chunk_type}: {count} chunks")
    
    def create_vector_database(self, output_prefix: str = "troubleshooting"):
        """Create vector database with better indexing"""
        if not self.chunks:
            print("No chunks to process. Please run process_directory() first.")
            return
        
        print("\nCreating embeddings for all chunks...")
        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype('float32'))
        
        # Create index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save everything
        index_path = f"{output_prefix}_index.bin"
        faiss.write_index(index, index_path)
        
        with open(f"{output_prefix}_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(f"{output_prefix}_metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        with open(f"{output_prefix}_patterns.pkl", 'wb') as f:
            pickle.dump(self.document_patterns, f)
        
        print(f"\nVector database created successfully!")
        print(f"Files saved: {index_path}, {output_prefix}_chunks.pkl, {output_prefix}_metadata.pkl")
    
    def test_search(self, query: str, top_k: int = 3):
        """Test search functionality"""
        try:
            index = faiss.read_index("troubleshooting_index.bin")
            with open("troubleshooting_chunks.pkl", 'rb') as f:
                chunks = pickle.load(f)
            with open("troubleshooting_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding.astype('float32'))
            scores, indices = index.search(query_embedding.astype('float32'), top_k)
            
            print(f"\nQuery: '{query}'")
            print("=" * 60)
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                meta = metadata[idx]
                print(f"\n{i+1}. Similarity: {score:.3f}")
                print(f"   Source: {meta['source']}")
                print(f"   Type: {meta['chunk_type']}")
                
                if meta.get('title') or meta.get('section_title'):
                    title = meta.get('title') or meta.get('section_title')
                    print(f"   Title/Section: {title[:80]}...")
                
                print(f"   Length: {meta.get('word_count', 0)} words")
                print(f"   Preview: {chunks[idx][:200]}...")
                
        except FileNotFoundError:
            print("Vector database files not found. Please run create_vector_database() first.")

def main():
    processor = AdaptiveTroubleshootingDocumentProcessor()
    
    print("This processor will analyze your documents to find their natural structure.")
    print("It will show you what patterns it finds and adapt accordingly.\n")
    
    # Process documents
    processor.process_directory("docs")
    
    if processor.chunks:
        print("\n" + "="*60)
        print("CREATING VECTOR DATABASE")
        print("="*60)
        processor.create_vector_database()
        
        print("\n" + "="*60)
        print("TESTING SEARCH FUNCTIONALITY")
        print("="*60)
        
        # Use more generic test queries
        test_queries = [
            "troubleshooting",
            "problem",
            "solution",
            "error",
            "repair"
        ]
        
        for query in test_queries:
            processor.test_search(query)
    else:
        print("No documents were processed.")

if __name__ == "__main__":
    main()