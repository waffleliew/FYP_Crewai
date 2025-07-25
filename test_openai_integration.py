#!/usr/bin/env python3
"""
Test script to verify OpenAI API integration with both autogenAI and crewAI modules.
"""

import os
from dotenv import load_dotenv

# Set AutoGen Docker configuration
os.environ["AUTOGEN_USE_DOCKER"] = "0"

from autogenAI import run_analysis as autogen_run_analysis
from crewAI import run_analysis as crew_run_analysis

# Load environment variables
load_dotenv()

def test_openai_integration():
    """Test OpenAI integration with both analysis modules."""
    
    # Check if OpenAI API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return False
    
    print("✅ OPENAI_API_KEY found")
    
    # Sample transcript for testing
    sample_transcript = """
    Q1 2024 Earnings Call Transcript
    
    CEO: "We're pleased to report strong Q1 results with revenue growth of 15% year-over-year."
    CFO: "Our gross margins improved to 45% from 42% in the previous quarter."
    CEO: "We see continued momentum in our core markets and expect 10-12% growth for the full year."
    """
    
    # Test OpenAI models
    openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    
    print("\n🧪 Testing OpenAI integration...")
    
    for model in openai_models:
        print(f"\n📊 Testing {model} with AutoGen...")
        try:
            result = autogen_run_analysis(
                transcript=sample_transcript,
                ticker="TEST",
                model=model,
                temp=0.7,
                verbose_mode=False,
                max_rounds=2  # Reduced for testing
            )
            print(f"✅ {model} AutoGen test completed successfully")
            print(f"   Result length: {len(result)} characters")
        except Exception as e:
            print(f"❌ {model} AutoGen test failed: {str(e)}")
        
        print(f"\n📊 Testing {model} with CrewAI...")
        try:
            result = crew_run_analysis(
                transcript=sample_transcript,
                ticker="TEST",
                model=model,
                temp=0.7,
                verbose_mode=False
            )
            print(f"✅ {model} CrewAI test completed successfully")
            print(f"   Result length: {len(result)} characters")
        except Exception as e:
            print(f"❌ {model} CrewAI test failed: {str(e)}")
    
    print("\n🎉 OpenAI integration testing completed!")
    return True

if __name__ == "__main__":
    test_openai_integration() 