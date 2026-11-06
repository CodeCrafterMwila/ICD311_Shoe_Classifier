import subprocess
import sys
from pathlib import Path

def main():
    venv_dir = Path("venv")
    
    # Check if virtual environment exists, if not create it
    if not venv_dir.exists():
        print("Creating virtual environment (.venv)... - run.py:10")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        
    # Determine correct pip and python paths based on OS
    if sys.platform == "win32":
        pip_path = venv_dir / "Scripts" / "pip.exe"
        streamlit_path = venv_dir / "Scripts" / "streamlit.exe"
    else:
        pip_path = venv_dir / "bin" / "pip"
        streamlit_path = venv_dir / "bin" / "streamlit"
        
    # Install or upgrade requirements
    print("Ensuring dependencies are installed... - run.py:22")
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
    
    # Run Streamlit
    print("Launching Streamlit App... - run.py:27")
    subprocess.run([str(streamlit_path), "run", "app.py"])

if __name__ == "__main__":
    main()