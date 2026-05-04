"""
Test script for Visual Assistant
Run this to verify all components are working
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    tests = [
        ("requests", "HTTP requests"),
        ("base64", "Image encoding"),
        ("json", "JSON parsing"),
        ("threading", "Multi-threading"),
        ("pygame", "Audio playback"),
        ("PIL", "Image processing"),
    ]
    
    failed = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"  ✓ {description} ({module})")
        except ImportError as e:
            print(f"  ✗ {description} ({module}): {e}")
            failed.append(module)
    
    return len(failed) == 0, failed


def test_core_module():
    """Test core module functionality"""
    print("\nTesting core module...")
    
    try:
        from core import (
            APIConfig,
            ImageEncoder,
            LlamaVisionService,
            OCRService,
            TextProcessor,
            TTSService,
            VisualAssistant
        )
        print("  ✓ All classes imported successfully")
        
        # Test API configuration
        assert hasattr(APIConfig, 'LLAMA_API_KEY'), "Missing LLAMA_API_KEY"
        assert hasattr(APIConfig, 'TTS_API_KEY'), "Missing TTS_API_KEY"
        assert hasattr(APIConfig, 'OCR_API_KEY'), "Missing OCR_API_KEY"
        assert hasattr(APIConfig, 'LLM_API_KEY'), "Missing LLM_API_KEY"
        print("  ✓ API configuration valid")
        
        # Test service initialization
        vision = LlamaVisionService()
        assert vision.api_key is not None, "Vision API key not set"
        print("  ✓ Vision service initialized")
        
        ocr = OCRService()
        assert ocr.api_key is not None, "OCR API key not set"
        print("  ✓ OCR service initialized")
        
        tts = TTSService()
        assert tts.api_key is not None, "TTS API key not set"
        print("  ✓ TTS service initialized")
        
        processor = TextProcessor()
        assert processor.api_key is not None, "LLM API key not set"
        print("  ✓ Text processor initialized")
        
        assistant = VisualAssistant()
        print("  ✓ Visual Assistant initialized")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Core module test failed: {e}")
        return False


def test_gui_module():
    """Test GUI module (without running it)"""
    print("\nTesting GUI module...")
    
    try:
        # Just check if file exists and has correct structure
        with open('gui.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'class VisualAssistantGUI' in content, "Main GUI class not found"
        assert 'def process_image' in content, "Process method not found"
        assert 'def load_image' in content, "Load image method not found"
        print("  ✓ GUI module structure valid")
        
        return True
        
    except Exception as e:
        print(f"  ✗ GUI module test failed: {e}")
        return False


def test_mobile_module():
    """Test mobile module"""
    print("\nTesting mobile module...")
    
    try:
        with open('mobile_app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'class VisualAssistantMobile' in content, "Mobile app class not found"
        assert 'from kivy.app import App' in content, "Kivy import not found"
        print("  ✓ Mobile module structure valid")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Mobile module test failed: {e}")
        return False


def test_api_connectivity():
    """Test API connectivity (optional, requires internet)"""
    print("\nTesting API connectivity...")
    
    try:
        import requests
        
        # Test NVIDIA API endpoint
        response = requests.get(
            "https://integrate.api.nvidia.com/health",
            timeout=5
        )
        
        if response.status_code == 200:
            print("  ✓ NVIDIA API endpoint reachable")
            return True
        else:
            print(f"  ⚠ NVIDIA API returned status {response.status_code}")
            return True  # Still okay
            
    except requests.exceptions.RequestException:
        print("  ⚠ Cannot reach NVIDIA API (offline or firewall)")
        return True  # Don't fail the test
    except Exception as e:
        print(f"  ⚠ Connectivity test error: {e}")
        return True


def run_all_tests():
    """Run all tests and report results"""
    print("="*60)
    print("VISUAL ASSISTANT - TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Core Module", (test_core_module(), [])))
    results.append(("GUI Module", (test_gui_module(), [])))
    results.append(("Mobile Module", (test_mobile_module(), [])))
    results.append(("API Connectivity", (test_api_connectivity(), [])))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, (success, _) in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print("-"*60)
    print(f"Total: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python gui.py' to launch the application")
        print("2. Or double-click 'run.bat' on Windows")
        print("3. Load an image and start analyzing!")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("You may still be able to use the application.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
