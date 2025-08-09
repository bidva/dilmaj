#!/usr/bin/env python3
"""
Simple test script to verify the PDF translator setup.
Use this with the "PDF Translator - Current File" debug configuration.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pdf_translator.config import Config
from pdf_translator.exceptions import ConfigurationError
from pdf_translator.utils import validate_api_key


def main():
    """Test the basic setup and configuration."""
    print("🔧 Testing PDF Translator Setup...")
    
    # Test 1: Check API Key
    try:
        api_key = validate_api_key()
        print("✅ OpenAI API key found and valid")
        # Don't print the actual key for security
        print(f"   Key starts with: {api_key[:8]}...")
    except ConfigurationError as e:
        print(f"❌ API Key Error: {e}")
        print("   💡 Make sure to set OPENAI_API_KEY in your .env file")
        return False
    
    # Test 2: Create a basic config
    try:
        config = Config(
            model="gpt-3.5-turbo",
            prompt="Translate this to English",
            verbose=True
        )
        print("✅ Configuration created successfully")
        print(f"   Model: {config.model}")
        print(f"   Rate limit: {config.rate_limit_rpm} RPM")
        print(f"   Concurrent requests: {config.concurrent_requests}")
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        return False
    
    # Test 3: Check dependencies
    try:
        from langchain_openai import ChatOpenAI
        from pypdf import PdfReader
        print("✅ Core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   💡 Run 'poetry install' to install dependencies")
        return False
    
    # Test 4: Check project structure
    expected_files = [
        "pdf_translator/__init__.py",
        "pdf_translator/cli.py",
        "pdf_translator/processor.py",
        "pdf_translator/config.py",
        "tests/test_config.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ Project structure looks good")
    
    print("\n🎉 Setup test completed successfully!")
    print("\n📖 Next steps:")
    print("1. Place a PDF file in the project directory (or use an absolute path)")
    print("2. Use the 'PDF Translator - Debug with Sample PDF' configuration")
    print("3. Set breakpoints in processor.py to debug the processing flow")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
