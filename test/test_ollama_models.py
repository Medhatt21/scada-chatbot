#!/usr/bin/env python3
"""
Test script for Ollama models and connectivity.
Run this to debug LLM model issues.
"""
import asyncio
import sys
import os
import requests
import json
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import settings


def test_ollama_connection():
    """Test basic Ollama connection."""
    print("🔍 Testing Ollama connection...")
    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/version", timeout=5)
        if response.status_code == 200:
            print(f"✅ Ollama is running - Version: {response.json()}")
            return True
        else:
            print(f"❌ Ollama responded with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        return False


def test_list_models():
    """Test listing available models."""
    print("\n🔍 Testing model listing...")
    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            print(f"✅ Models API response: {json.dumps(models_data, indent=2)}")
            
            models = models_data.get('models', [])
            if models:
                print(f"📋 Available models ({len(models)}):")
                for model in models:
                    model_name = model.get('name', 'Unknown')
                    model_size = model.get('size', 'Unknown size')
                    print(f"  - {model_name} ({model_size} bytes)")
                return models
            else:
                print("❌ No models found")
                return []
        else:
            print(f"❌ Failed to list models: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to list models: {e}")
        return []


def test_model_availability(model_name: str):
    """Test if a specific model is available."""
    print(f"\n🔍 Testing model availability: {model_name}")
    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get('models', [])
            
            for model in models:
                if model.get('name') == model_name:
                    print(f"✅ Model {model_name} is available")
                    return True
            
            print(f"❌ Model {model_name} not found")
            print("Available models:")
            for model in models:
                print(f"  - {model.get('name')}")
            return False
        else:
            print(f"❌ Failed to check model: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to check model: {e}")
        return False


def test_llm_generation(model_name: str):
    """Test LLM text generation."""
    print(f"\n🔍 Testing LLM generation with {model_name}...")
    try:
        payload = {
            "model": model_name,
            "prompt": "Hello! Please respond with a simple greeting.",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 50
            }
        }
        
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            print(f"✅ LLM generation successful:")
            print(f"📝 Response: {generated_text}")
            return True
        else:
            print(f"❌ LLM generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ LLM generation failed: {e}")
        return False


def test_embedding_generation(model_name: str):
    """Test embedding generation."""
    print(f"\n🔍 Testing embedding generation with {model_name}...")
    try:
        payload = {
            "model": model_name,
            "prompt": "This is a test document for embedding generation."
        }
        
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/embeddings",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            embedding = result.get('embedding', [])
            print(f"✅ Embedding generation successful:")
            print(f"📊 Embedding dimensions: {len(embedding)}")
            print(f"📊 First 5 values: {embedding[:5]}")
            return True
        else:
            print(f"❌ Embedding generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Embedding generation failed: {e}")
        return False


def test_model_pull(model_name: str):
    """Test pulling a model if it's not available."""
    print(f"\n🔍 Testing model pull: {model_name}")
    try:
        payload = {"name": model_name}
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/pull",
            json=payload,
            timeout=300  # 5 minutes for pulling
        )
        
        if response.status_code == 200:
            print(f"✅ Model {model_name} pulled successfully")
            return True
        else:
            print(f"❌ Failed to pull model: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to pull model: {e}")
        return False


def main():
    """Run all Ollama tests."""
    print("🚀 Starting Ollama Model Tests")
    print("=" * 50)
    
    print(f"📋 Configuration:")
    print(f"  - Ollama URL: {settings.OLLAMA_BASE_URL}")
    print(f"  - LLM Model: {settings.OLLAMA_LLM_MODEL}")
    print(f"  - Embedding Model: {settings.OLLAMA_EMBEDDING_MODEL}")
    
    # Test connection
    if not test_ollama_connection():
        print("\n❌ Cannot connect to Ollama. Please ensure it's running.")
        return False
    
    # List available models
    models = test_list_models()
    
    # Test LLM model
    print("\n" + "=" * 50)
    print("Testing LLM Model")
    print("=" * 50)
    
    if not test_model_availability(settings.OLLAMA_LLM_MODEL):
        print(f"⚠️  Attempting to pull {settings.OLLAMA_LLM_MODEL}...")
        if not test_model_pull(settings.OLLAMA_LLM_MODEL):
            print(f"❌ Failed to pull {settings.OLLAMA_LLM_MODEL}")
            return False
    
    if not test_llm_generation(settings.OLLAMA_LLM_MODEL):
        print(f"❌ LLM generation test failed for {settings.OLLAMA_LLM_MODEL}")
        return False
    
    # Test embedding model
    print("\n" + "=" * 50)
    print("Testing Embedding Model")
    print("=" * 50)
    
    if not test_model_availability(settings.OLLAMA_EMBEDDING_MODEL):
        print(f"⚠️  Attempting to pull {settings.OLLAMA_EMBEDDING_MODEL}...")
        if not test_model_pull(settings.OLLAMA_EMBEDDING_MODEL):
            print(f"❌ Failed to pull {settings.OLLAMA_EMBEDDING_MODEL}")
            return False
    
    if not test_embedding_generation(settings.OLLAMA_EMBEDDING_MODEL):
        print(f"❌ Embedding generation test failed for {settings.OLLAMA_EMBEDDING_MODEL}")
        return False
    
    print("\n🎉 All Ollama tests passed!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 