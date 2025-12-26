#!/usr/bin/env python
"""Check Weaviate collection status and data."""

import weaviate

# Connect to Weaviate
client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
)

try:
    # Get collection
    chunks = client.collections.get("CodeChunk")
    
    # Get collection metadata
    result = chunks.query.fetch_objects(limit=1)
    
    print(f"✓ Connected to Weaviate")
    print(f"✓ CodeChunk collection exists")
    
    # Count objects
    count_result = chunks.query.fetch_objects(limit=1)
    if count_result.objects:
        print(f"✓ Collection has data")
        
        # Show first object
        obj = count_result.objects[0]
        print(f"\n--- First Object ---")
        print(f"UUID: {obj.uuid}")
        print(f"File: {obj.properties.get('file_path')}")
        print(f"Language: {obj.properties.get('language')}")
        print(f"Lines: {obj.properties.get('start_line')}-{obj.properties.get('end_line')}")
        print(f"Repository: {obj.properties.get('repository')}")
        if obj.vector:
            print(f"Vector: {len(obj.vector)} dimensions")
        print(f"\nContent preview:")
        content = obj.properties.get('content', '')
        print(content[:200] + "..." if len(content) > 200 else content)
    else:
        print("✗ Collection is empty!")
        
    # Try to get total count
    print("\n--- Attempting to count all objects ---")
    all_objects = chunks.query.fetch_all()
    print(f"Total objects in collection: {len(all_objects.objects)}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    client.close()
