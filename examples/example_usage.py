"""
Example usage of the Multimodal RAG Engine
"""

import os
from pathlib import Path

# Set up environment
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

from src.processor.pdf_parser import PDFParser
from src.processor.element_classifier import ElementClassifier
from src.processor.image_describer import ImageDescriber
from src.processor.table_extractor import TableExtractor
from src.indexing.vectorstore import VectorStore
from src.retrieval.searcher import Searcher
from src.generation.generator import Generator

def main():
    # Initialize components
    parser = PDFParser()
    classifier = ElementClassifier()
    describer = ImageDescriber()
    table_extractor = TableExtractor()
    vectorstore = VectorStore()
    searcher = Searcher(vectorstore)
    generator = Generator()
    
    # Process a document
    print("📄 Processing document...")
    pdf_path = "examples/sample.pdf"  # Replace with your PDF
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    elements = parser.parse(pdf_path)
    print(f"✅ Extracted {len(elements)} elements")
    
    # Process each element
    print("🔍 Processing elements...")
    for element in elements:
        element_type = classifier.classify(element)
        
        if element_type == "image" and element.get("content"):
            print(f"  - Describing image on page {element['page']}")
            element["description"] = describer.describe(element["content"])
        
        elif element_type == "table":
            print(f"  - Extracting table on page {element['page']}")
            table_data = table_extractor.extract(element.get("content", {}))
            element["content"] = table_data
    
    # Index elements
    print("💾 Indexing elements...")
    vectorstore.index_elements(elements)
    print("✅ Indexing complete")
    
    # Query examples
    queries = [
        "What are the main findings?",
        "What does the chart show?",
        "Summarize the data in the table"
    ]
    
    print("\n🔎 Running queries...")
    for query in queries:
        print(f"\nQ: {query}")
        results = searcher.search(query, top_k=3)
        response = generator.generate(query, results)
        print(f"A: {response['answer']}")
        print(f"Sources: {len(response['sources'])} elements")

if __name__ == "__main__":
    main()
