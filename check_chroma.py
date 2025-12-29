#!/usr/bin/env python3
"""
Script to inspect ChromaDB contents
"""
import os
from langchain_chroma import Chroma
from vectors.embeddings_model import embeddings_model

def check_chroma_db(persist_dir="./chroma_db", collection_name="pdf"):
    """Check the contents of a ChromaDB collection"""
    
    if not os.path.exists(persist_dir):
        print(f"❌ ChromaDB directory does not exist: {persist_dir}")
        return
    
    print(f"✅ ChromaDB directory exists: {persist_dir}")
    print(f"📂 Files in directory:")
    for file in os.listdir(persist_dir):
        file_path = os.path.join(persist_dir, file)
        size = os.path.getsize(file_path)
        print(f"   - {file} ({size:,} bytes)")
    
    print(f"\n🔍 Loading collection '{collection_name}'...")
    
    try:
        # Load the embeddings model
        embeddings = embeddings_model()
        
        # Connect to the existing ChromaDB
        vectorstore = Chroma(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        # Get collection info
        collection = vectorstore._collection
        count = collection.count()
        
        print(f"✅ Collection loaded successfully!")
        print(f"📊 Number of documents/chunks: {count}")
        
        if count > 0:
            # Get a sample document
            print(f"\n📄 Sample documents:")
            results = vectorstore.similarity_search("", k=3)
            for i, doc in enumerate(results[:3], 1):
                content_preview = doc.page_content[:200].replace('\n', ' ')
                print(f"\n   Document {i}:")
                print(f"   Content: {content_preview}...")
                print(f"   Metadata: {doc.metadata}")
            
            # Try a sample query
            print(f"\n🔎 Testing search functionality...")
            query = "experience"  # Change this to match your resume content
            search_results = vectorstore.similarity_search(query, k=2)
            print(f"   Query: '{query}'")
            print(f"   Results found: {len(search_results)}")
            if search_results:
                print(f"   Top result preview: {search_results[0].page_content[:150]}...")
        else:
            print("⚠️  Collection is empty!")
            
    except Exception as e:
        print(f"❌ Error loading collection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    persist_dir = sys.argv[1] if len(sys.argv) > 1 else "./chroma_db"
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "pdf"
    
    print("="*60)
    print("ChromaDB Inspector")
    print("="*60)
    check_chroma_db(persist_dir, collection_name)
    print("="*60)
