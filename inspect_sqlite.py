#!/usr/bin/env python3
"""
SQLite3 database inspector using Python's built-in sqlite3 module
"""
import sqlite3
import sys

def inspect_sqlite_db(db_path):
    """Inspect a SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"📊 Database: {db_path}")
        print("=" * 60)
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"\n📋 Tables ({len(tables)}):")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   - {table_name}: {count:,} rows")
        
        # Show collections
        print(f"\n🗂️  Collections:")
        try:
            cursor.execute("SELECT id, name FROM collections")
            collections = cursor.fetchall()
            for coll_id, coll_name in collections:
                print(f"   - {coll_name} (ID: {coll_id})")
        except sqlite3.OperationalError:
            print("   No collections table found")
        
        # Show embeddings count
        print(f"\n🎯 Embeddings:")
        try:
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            emb_count = cursor.fetchone()[0]
            print(f"   Total embeddings: {emb_count:,}")
            
            if emb_count > 0:
                cursor.execute("SELECT * FROM embeddings LIMIT 1")
                columns = [description[0] for description in cursor.description]
                print(f"   Columns: {', '.join(columns)}")
        except sqlite3.OperationalError:
            print("   No embeddings table found")
        
        # Show segments
        print(f"\n📦 Segments:")
        try:
            cursor.execute("SELECT COUNT(*) FROM segments")
            seg_count = cursor.fetchone()[0]
            print(f"   Total segments: {seg_count:,}")
        except sqlite3.OperationalError:
            print("   No segments table found")
        
        conn.close()
        print("=" * 60)
        print("✅ Inspection complete!")
        
    except sqlite3.Error as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./chroma_db/chroma.sqlite3"
    inspect_sqlite_db(db_path)
