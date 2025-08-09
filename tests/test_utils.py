"""Tests for the utils module."""

import os
import pytest

from pdf_translator.utils import validate_api_key
from pdf_translator.exceptions import ConfigurationError


class TestUtils:
    """Test cases for utility functions."""
    
    def test_validate_api_key_valid(self):
        """Test validation with a valid API key."""
        # Temporarily set a valid-looking API key
        original_key = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = 'sk-test1234567890abcdef1234567890abcdef123456'
        
        try:
            result = validate_api_key()
            assert result == 'sk-test1234567890abcdef1234567890abcdef123456'
        finally:
            # Restore original key
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
            elif 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
    
    def test_validate_api_key_missing(self):
        """Test validation when API key is missing."""
        # Temporarily remove API key
        original_key = os.environ.get('OPENAI_API_KEY')
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        try:
            with pytest.raises(ConfigurationError) as excinfo:
                validate_api_key()
            assert "OPENAI_API_KEY not found" in str(excinfo.value)
        finally:
            # Restore original key
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
    
    def test_validate_api_key_empty(self):
        """Test validation with empty API key."""
        original_key = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = '   '  # Just whitespace
        
        try:
            with pytest.raises(ConfigurationError) as excinfo:
                validate_api_key()
            assert "is empty" in str(excinfo.value)
        finally:
            # Restore original key
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
            elif 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
    
    def test_validate_api_key_placeholder(self):
        """Test validation with placeholder API key."""
        placeholders = [
            'your_openai_api_key_here',
            'your_api_key_here',
            'sk-your-api-key-here',
            'replace_with_your_api_key',
            'your-openai-api-key',
            'put_your_api_key_here'
        ]
        
        original_key = os.environ.get('OPENAI_API_KEY')
        
        for placeholder in placeholders:
            os.environ['OPENAI_API_KEY'] = placeholder
            
            try:
                with pytest.raises(ConfigurationError) as excinfo:
                    validate_api_key()
                assert "placeholder value" in str(excinfo.value)
                assert placeholder in str(excinfo.value)
            finally:
                pass  # Clean up after all tests
        
        # Final cleanup
        if original_key:
            os.environ['OPENAI_API_KEY'] = original_key
        elif 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
    
    def test_validate_api_key_strips_whitespace(self):
        """Test that API key validation strips whitespace."""
        original_key = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = '  sk-test1234567890abcdef1234567890abcdef123456  '
        
        try:
            result = validate_api_key()
            assert result == 'sk-test1234567890abcdef1234567890abcdef123456'
        finally:
            # Restore original key
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
            elif 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
    
    def test_validate_api_key_warns_invalid_format(self, caplog):
        """Test that validation warns for keys not starting with 'sk-'."""
        original_key = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = 'invalid-key-format-12345678901234567890'
        
        try:
            result = validate_api_key()
            assert result == 'invalid-key-format-12345678901234567890'
            assert "does not start with 'sk-'" in caplog.text
        finally:
            # Restore original key
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
            elif 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
    
    def test_validate_api_key_warns_short_key(self, caplog):
        """Test that validation warns for unusually short keys."""
        original_key = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = 'sk-short'
        
        try:
            result = validate_api_key()
            assert result == 'sk-short'
            assert "appears unusually short" in caplog.text
        finally:
            # Restore original key
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
            elif 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
