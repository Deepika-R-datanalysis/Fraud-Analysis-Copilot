"""Step 1 check: confirms your libraries are installed."""
import sys

print("Python:", sys.version.split()[0])
for name in ["pandas", "numpy", "sklearn", "duckdb"]:
    try:
        module = __import__(name)
        print(f"OK  {name} {getattr(module, '__version__', '')}")
    except ImportError:
        print(f"MISSING  {name}  -> run: pip install {name}")
