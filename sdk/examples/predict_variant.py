#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genos import create_client
from genos.exceptions import APIRequestError, ValidationError

def main():
    # Create client with your API token
    client = create_client(token="your_token_here")
    
    print("🧬 Genos Variant Prediction Example")
    print("=" * 50)
    
    # Example 1: Basic variant prediction with hg19
    print("\n📊 Example 1: Basic variant prediction (hg19)")
    try:
        result = client.variant_predict(
            assembly="hg19",
            chrom="chr6", 
            pos=51484075,
            ref="T",
            alt="G"
        )
        print(f"✅ Prediction: {result['prediction']}")
        print(f"   Pathogenic Score: {result['score_Pathogenic']:.4f}")
        print(f"   Benign Score: {result['score_Benign']:.4f}")
    except APIRequestError as e:
        print(f"❌ API Error: {e}")
    except ValidationError as e:
        print(f"❌ Validation Error: {e}")
    
    # Example 2: Using hg38 reference genome
    print("\n📊 Example 2: hg38 reference genome")
    try:
        result = client.variant_predict(
            assembly="hg38",
            chrom="chr6",
            pos=51484075, 
            ref="T",
            alt="G"
        )
        print(f"✅ Prediction: {result['prediction']}")
        print(f"   Pathogenic Score: {result['score_Pathogenic']:.4f}")
        print(f"   Benign Score: {result['score_Benign']:.4f}")
    except APIRequestError as e:
        print(f"❌ API Error: {e}")
    except ValidationError as e:
        print(f"❌ Validation Error: {e}")
    
    # Example 3: Error handling demonstration
    print("\n📊 Example 3: Error handling")
    try:
        # This will cause a validation error
        result = client.variant_predict(
            assembly="invalid_assembly",
            chrom="chr6",
            pos=51484075,
            ref="T", 
            alt="G"
        )
    except ValidationError as e:
        print(f"✅ Caught expected validation error: {e}")

if __name__ == "__main__":
    main()

