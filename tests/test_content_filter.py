import pytest
from src.section_7_ethics.content_filter import ContentFilter, filter_llm_output

def test_content_filter():
    # Initialize content filter
    content_filter = ContentFilter()
    
    # Test safe content
    safe_text = "This is a nice message."
    assert content_filter.is_safe(safe_text), "Safe content incorrectly flagged as unsafe"
    
    # Test unsafe content
    unsafe_text = "I hate you and wish you harm!"
    assert not content_filter.is_safe(unsafe_text), "Unsafe content incorrectly flagged as safe"
    
    # Test filter_llm_output
    filtered_output = filter_llm_output(safe_text)
    assert filtered_output == safe_text, "Safe content modified unexpectedly"
    
    filtered_output = filter_llm_output(unsafe_text)
    assert "blocked" in filtered_output.lower(), "Unsafe content not blocked"
