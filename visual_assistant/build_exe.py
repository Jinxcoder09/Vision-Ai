"""
Build script for creating Windows executable
Run this to create a standalone .exe file
"""

import os
import sys
import subprocess
import shutil

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    try:
        import PyInstaller
        print("✓ PyInstaller found")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    print("All dependencies ready!")

def build_executable():
    """Build the Windows executable"""
    
    print("\n" + "="*60)
    print("Building Visual Assistant Executable")
    print("="*60 + "\n")
    
    # Clean previous builds
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')
    
    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name=VisualAssistant",
        "--icon=NONE",
        "--add-data=core.py;.",
        "--hidden-import=pkg_resources.py2_warn",
        "gui.py"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        subprocess.check_call(cmd)
        
        print("\n" + "="*60)
        print("✓ Build Successful!")
        print("="*60)
        print(f"\nExecutable created at: dist/VisualAssistant.exe")
        print("\nTo distribute:")
        print("1. Copy VisualAssistant.exe to target computer")
        print("2. Ensure Python is installed (or include it)")
        print("3. Run the executable")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Build failed with error: {e}")
        return False
    
    return True

def create_distribution_package():
    """Create a distribution package with all necessary files"""
    
    print("\nCreating distribution package...")
    
    dist_dir = "distribution"
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    
    os.makedirs(dist_dir)
    
    # Copy executable
    if os.path.exists('dist/VisualAssistant.exe'):
        shutil.copy('dist/VisualAssistant.exe', dist_dir)
    
    # Copy supporting files
    files_to_copy = [
        'README.md',
        'requirements.txt',
        'run.bat'
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy(file, dist_dir)
    
    # Create install script
    install_script = """@echo off
echo ================================================
echo   Visual Assistant - Setup
echo ================================================
echo.

REM Install dependencies
echo Installing required packages...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Installation complete!
echo You can now run VisualAssistant.exe
pause
"""
    
    with open(os.path.join(dist_dir, 'install.bat'), 'w') as f:
        f.write(install_script)
    
    print(f"Distribution package created in '{dist_dir}' folder")

def main():
    """Main build process"""
    
    print("Visual Assistant - Build Script")
    print("="*60)
    
    # Check dependencies
    check_dependencies()
    
    # Build executable
    if build_executable():
        # Create distribution package
        create_distribution_package()
        
        print("\n" + "="*60)
        print("🎉 All done!")
        print("="*60)
        print("\nNext steps:")
        print("1. Test the executable: dist/VisualAssistant.exe")
        print("2. Distribute the 'distribution' folder")
        print("3. For Android APK, see android_build_guide.md")
    else:
        print("\nBuild failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
