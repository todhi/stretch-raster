"""
Stretch v2.0 - Main Entry Point
"""
import sys
import os

# Ensure current directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stretch import RasterHistGUI

def main():
    app = RasterHistGUI()
    app.run()

if __name__ == "__main__":
    main()
