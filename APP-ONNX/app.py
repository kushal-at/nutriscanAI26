"""
APP/app.py - NutriScan Main Application Entry Point
===================================================
Serves the BackEnd gateway on Port 8000 (connecting to FrontEnd,
DL on Port 8001, and ML on Port 8002).
"""

from BackEnd.app import app

__all__ = ["app"]
