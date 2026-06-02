"""
TUBITAK 2209-A - Pnomotoraks AI Demo
Kullanim: python demo/demo_calistir.py
"""
import subprocess, sys, time, webbrowser, os, requests
from pathlib import Path

API    = "http://localhost:8000"
UVICORN = sys.executable.replace("python3", "uvicorn").replace("python", "uvicorn")

def api_hazir():
    try:
        return requests.get(f"{API}/health", timeout=2).status_code == 200
    except:
        return False

def api_baslat():
    root = Path(__file__).parent.parent
    if api_hazir():
        print("API zaten calisiyor.")
        return None
    print("API baslatiliyor...", end=" ", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for _ in range(20):
        time.sleep(1)
        if api_hazir():
            print("hazir!")
            return proc
        print(".", end="", flush=True)
    print("\nHATA: API baslanamadi. 'pip install -r requirements.txt' calistirin.")
    sys.exit(1)

def main():
    print("=" * 50)
    print("  PNOMOTORAKS AI - DEMO")
    print("  TUBITAK 2209-A")
    print("=" * 50)
    print()

    proc = api_baslat()

    health = requests.get(f"{API}/health").json()
    fold_sayisi = sum(health.get("folds", {}).values())
    print(f"Model: {fold_sayisi}-fold ensemble | Cihaz: {health.get('device','?').upper()}")
    print()

    goruntu_dir = Path(__file__).parent / "hastalar"
    goruntular  = sorted(goruntu_dir.glob("*.png"))

    print(f"{len(goruntular)} demo goruntusu hazir:")
    for g in goruntular:
        kaynak = "VERİ (DX)" if g.name.startswith("veri") else "TUBITAK (CR)"
        print(f"  - {g.name:35s} [{kaynak}]")

    print()
    print("Tarayici aciliyor: http://localhost:8000")
    print("Goruntuler 'demo/hastalar/' klasorunde - surukle birak veya yukle.")
    print()
    print("Durdurmak icin Ctrl+C")

    webbrowser.open("http://localhost:8000")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDurduruluyor...")
        if proc:
            proc.terminate()

if __name__ == "__main__":
    main()
