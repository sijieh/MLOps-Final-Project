#!/usr/bin/env python3

import subprocess
import sys
import os

def install_dependencies():
    required_packages = ["fastapi", "uvicorn[standard]", "pydantic"]
    for package in required_packages:
        try:
            __import__(package.split('[')[0])
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def run_app():

    print("Access: http://localhost:8000")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    install_dependencies()
    run_app()