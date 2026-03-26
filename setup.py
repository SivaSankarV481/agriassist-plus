"""
setup.py
=========
One-click setup for AgriAssist+.
Run this ONCE to install dependencies, generate dataset, and train models.

    python setup.py
"""

import subprocess
import sys
import os

def run(cmd, desc):
    print(f"\n{'='*50}")
    print(f"  {desc}")
    print(f"{'='*50}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  ❌ Failed: {cmd}")
        sys.exit(1)
    print(f"  ✅ Done")

def main():
    print("\n🌾 AgriAssist+ Setup")
    print("=" * 50)

    # 1. Install dependencies
    run(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python packages...")

    # 2. Copy .env if not exists
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            print("\n  📝 .env file created from .env.example")
            print("  ⚠️  Edit .env and add your OPENROUTER_API_KEY for full LLM features")
            print("     (App works without it using rule-based recommendations)")

    # 3. Generate dataset
    run(f"{sys.executable} generate_dataset.py", "Generating Tamil Nadu crop dataset...")

    # 4. Train models
    run(f"{sys.executable} train_models.py", "Training XGBoost prediction models...")

    print("\n" + "=" * 50)
    print("  🎉 Setup Complete!")
    print("=" * 50)
    print("\n  Run the app:")
    print("    streamlit run app.py")
    print("\n  For full AI recommendations:")
    print("    1. Get free API key from https://openrouter.ai")
    print("    2. Add to .env: OPENROUTER_API_KEY=your_key_here")
    print("=" * 50)

if __name__ == "__main__":
    main()
