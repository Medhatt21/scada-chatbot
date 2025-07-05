#!/usr/bin/env python3
"""
Test runner script for Ollama and RAG pipeline tests.
"""
import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_test(test_script, description):
    """Run a specific test script."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, test_script],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run Ollama and RAG pipeline tests")
    parser.add_argument(
        "--test", 
        choices=["ollama", "rag", "all"], 
        default="all",
        help="Which test suite to run"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (skip model pulling)"
    )
    
    args = parser.parse_args()
    
    # Get test directory
    test_dir = Path(__file__).parent
    
    # Define test suites
    tests = []
    
    if args.test in ["ollama", "all"]:
        tests.append((
            test_dir / "test_ollama_models.py",
            "Ollama Models and Connectivity Tests"
        ))
    
    if args.test in ["rag", "all"]:
        tests.append((
            test_dir / "test_rag_pipeline.py",
            "RAG Pipeline and Embeddings Tests"
        ))
    
    if not tests:
        print("No tests selected")
        return False
    
    # Run tests
    print("🚀 Starting Test Suite")
    print(f"Running {len(tests)} test suite(s)...")
    
    results = []
    for test_script, description in tests:
        if not test_script.exists():
            print(f"❌ Test script not found: {test_script}")
            results.append((description, False))
            continue
            
        success = run_test(str(test_script), description)
        results.append((description, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Suite Summary")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description}: {status}")
    
    print(f"\n📊 Overall Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All test suites passed!")
        return True
    else:
        print("⚠️  Some test suites failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 