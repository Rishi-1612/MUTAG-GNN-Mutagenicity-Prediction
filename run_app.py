import os
import sys
import webbrowser
import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("  MUTAG-GNN: Molecular Mutagenicity Graph Neural Network")
    print("  Unified Full-Stack Prototype & Academic Defense Platform")
    print("=" * 65)
    print("\nStarting server on http://127.0.0.1:8000 ...")
    print("-> API documentation: http://127.0.0.1:8000/docs")
    print("-> Interactive Web App: http://127.0.0.1:8000/")
    print("\nPress Ctrl+C to terminate.")

    # Automatically open browser
    try:
        webbrowser.open("http://127.0.0.1:8000")
    except Exception:
        pass

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
