#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genos import create_client
from genos.exceptions import APIRequestError, PackageUpdateRequired, ValidationError

def main():
    # Create client with your API token
    client = create_client(token="your_token_here")
    
    print("🧬 Genos Variant Prediction Example")
    print("=" * 50)
    
    # Example 1: GRCh38 variant. score is the published fusion score (p_fusion).
    print("\n📊 Example 1: Basic variant prediction (hg38)")
    try:
        response = client.variant_predict(
            assembly="hg38",
            chrom="chr6",
            pos=51484075,
            ref="T",
            alt="G"
        )
        result = response["result"]
        print(f"✅ Prediction: {result['prediction']}")
        print(f"   Published score: {result['score']:.4f}")
        if result.get("p_mlp") is not None:
            print(f"   p_mlp: {result['p_mlp']:.4f}")
        if result.get("p_evee") is not None:
            print(f"   p_evee: {result['p_evee']:.4f}")
    except PackageUpdateRequired as e:
        print(f"❌ Package update required: {e}")
    except APIRequestError as e:
        print(f"❌ API Error: {e}")
    except ValidationError as e:
        print(f"❌ Validation Error: {e}")
    
    # Example 2: Using hg38 reference genome
    print("\n📊 Example 2: hg38 reference genome")
    try:
        response = client.variant_predict(
            assembly="hg38",
            chrom="chrX",
            pos=53580905,
            ref="T",
            alt="C"
        )
        result = response["result"]
        print(f"✅ Prediction: {result['prediction']}")
        print(f"   Published score: {result['score']:.4f}")
        if result.get("p_mlp") is not None:
            print(f"   p_mlp: {result['p_mlp']:.4f}")
        if result.get("p_evee") is not None:
            print(f"   p_evee: {result['p_evee']:.4f}")
    except PackageUpdateRequired as e:
        print(f"❌ Package update required: {e}")
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
    except (ValidationError, ValueError) as e:
        print(f"✅ Caught expected validation error: {e}")

if __name__ == "__main__":
    main()

