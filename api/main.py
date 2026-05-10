"""
api/main.py

Main entry point for the FastAPI application.
Run via: uvicorn api.main:app --reload
"""

from api.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
