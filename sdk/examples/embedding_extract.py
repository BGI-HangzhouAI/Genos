#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genos import create_client
from genos.exceptions import APIRequestError, ValidationError

def main():
    # Create client with your API token
    client = create_client(token="your_token_here")
    
    print("🧬 Genos Embedding Extraction Example")
    print("=" * 50)
    
    # Example 1: Basic embedding extraction
    print("\n📊 Example 1: Basic embedding extraction")
    try:
        sequence = "ATCGATCGATCGATCGATCGATCGATCG"
        result = client.get_embedding(sequence)
        
        print(f"✅ Sequence: {sequence}")
        print(f"   Length: {result['sequence_length']}")
        print(f"   Token Count: {result['token_count']}")
        print(f"   Embedding Dimension: {result['embedding_dim']}")
        print(f"   Embedding Shape: {result['embedding_shape']}")
        print(f"   Embedding Vector Length: {len(result['embedding'])}")
        
    except APIRequestError as e:
        print(f"❌ API Error: {e}")
    except ValidationError as e:
        print(f"❌ Validation Error: {e}")
    
    # Example 2: Different models
    print("\n📊 Example 2: Different models")
    models = ["Genos-1.2B", "Genos-10B"]
    for model in models:
        try:
            result = client.get_embedding(sequence, model_name=model)
            print(f"✅ Model {model}: {result['embedding_dim']} dimensions")
        except APIRequestError as e:
            print(f"❌ Model {model} Error: {e}")
        except ValidationError as e:
            print(f"❌ Model {model} Validation Error: {e}")
    
    # Example 3: Different pooling methods
    print("\n📊 Example 3: Different pooling methods")
    pooling_methods = ["mean", "max", "last", "none"]
    for method in pooling_methods:
        try:
            result = client.get_embedding(sequence, pooling_method=method)
            print(f"✅ Pooling {method}: {result['embedding_dim']} dimensions")
        except APIRequestError as e:
            print(f"❌ Pooling {method} Error: {e}")
        except ValidationError as e:
            print(f"❌ Pooling {method} Validation Error: {e}")
    
    # Example 4: Error handling
    print("\n📊 Example 4: Error handling")
    try:
        # Empty sequence should cause validation error
        result = client.get_embedding("")
    except ValidationError as e:
        print(f"✅ Caught expected validation error: {e}")

if __name__ == "__main__":
    main()