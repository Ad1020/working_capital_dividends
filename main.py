"""
Entry point for the Advanced Programming project.
This script verifies that the project repository runs correctly.
"""

from pathlib import Path

def main():
    print("Working Capital Dynamics and Dividend Policy")
    print("main.py executed successfully.\n")

    figures_dir = Path("figures")

    if figures_dir.exists():
        files = sorted(figures_dir.iterdir())
        print(f"Found {len(files)} file(s) in figures/:")
        for f in files:
            print(" -", f.name)
    else:
        print("figures/ directory not found.")

if __name__ == "__main__":
    main()

