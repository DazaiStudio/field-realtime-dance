import os
import subprocess
import time
import sys
import urllib.request
from urllib.error import URLError

def check_requirements():
    print("Checking backend dependencies...")
    # Using the root requirements.txt for convenience
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Checking launcher dependencies...")
    try:
        import webview
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pywebview"])

def ensure_model():
    model_path = 'pose_landmarker_full.task'
    if not os.path.exists(model_path):
        print(f"Downloading {model_path} (approx. 10MB)...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")

def start_backend():
    env = os.environ.copy()
    env.setdefault("FIELD_OSC_HOST", "127.0.0.1")
    env.setdefault("FIELD_OSC_PORT", "9000")
    env.setdefault("FIELD_OSC_ENABLED", "1")
    env.setdefault("FIELD_OSC_MODE", "raw")
    env.setdefault("FIELD_OSC_ALPHA", "1.0")
    print("Starting FastAPI backend on http://127.0.0.1:8000 ...")
    print(
        "OSC default -> "
        f"{env['FIELD_OSC_HOST']}:{env['FIELD_OSC_PORT']} "
        f"enabled={env['FIELD_OSC_ENABLED']} "
        f"mode={env['FIELD_OSC_MODE']} "
        f"alpha={env['FIELD_OSC_ALPHA']}"
    )
    # Path to app.py is inside backend folder
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd="backend",
        env=env
    )
    return backend_process

def serve_frontend_static():
    print("Serving static frontend files on http://127.0.0.1:5173 ...")
    if not os.path.exists("frontend/dist"):
        print("Building frontend, please wait (this may take a minute)...")
        subprocess.check_call("npm install && npm run build", cwd="frontend", shell=True)
        
    # Using http.server as a simple static file server
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", "5173"],
        cwd="frontend/dist"
    )
    return frontend_process

def wait_for_server(url, timeout=15):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url) as response:
                if response.getcode() == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def main():
    print("=== Realtime Dance Aesthetics Launcher ===")
    
    # 1. Environment Preparation
    check_requirements()
    ensure_model()
    
    # Import webview after pip install check
    import webview
    
    # 2. Start Services
    backend = start_backend()
    frontend = serve_frontend_static()
    
    print("Waiting for servers to initialize...")
    if not wait_for_server("http://127.0.0.1:5173"):
        print("Error: Frontend server failed to start.")
        backend.terminate()
        frontend.terminate()
        sys.exit(1)
        
    # 3. Open Application Window
    try:
        print("Opening Application Window...")
        window = webview.create_window(
            'Realtime Dance Aesthetics Analysis', 
            'http://127.0.0.1:5173',
            width=1400, 
            height=900,
            background_color='#0F172A',
            resizable=True
        )
        webview.start()
    except Exception as e:
        print(f"Error starting UI: {e}")
    finally:
        print("Shutting down servers...")
        backend.terminate()
        frontend.terminate()
        print("Goodbye!")

if __name__ == '__main__':
    main()
