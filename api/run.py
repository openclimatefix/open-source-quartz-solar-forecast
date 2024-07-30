import os
import sys
import uvicorn

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="localhost", port=8000, reload=True)