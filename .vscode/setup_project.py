from pathlib import Path

# Root folder: current directory (churn/)
root = Path.cwd()

# Define folder structure
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "app"
]

# Define placeholder files to create
files = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_model.ipynb",
    "src/preprocess.py",
    "src/train_model.py",
    "src/predict.py",
    "app/streamlit_app.py",
    "requirements.txt",
    "README.md",
    ".gitignore"
]

# Create folders
for folder in folders:
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)

# Create empty files
for file in files:
    path = root / file
    if not path.exists():
        path.touch()
        print(f"Created: {path.relative_to(root)}")

print("\n✅ Project structure created successfully!")
