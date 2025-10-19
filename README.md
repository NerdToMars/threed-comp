# 3D Object Reconstruction and Classification

This project demonstrates 3D object reconstruction and classification using the ModelNet10 dataset.

## 🚀 Quick Start with Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NerdToMars/threed-comp/blob/main/chongtian_hanzhang.ipynb)

Click the button above to open this notebook in Google Colab!

## 📊 Dataset

The project uses the ModelNet10 dataset, which contains 4,899 pre-aligned 3D shapes from 10 categories:

- 3,991 shapes for training (80%)
- 908 shapes for testing (20%)

The CAD models are in Object File Format (OFF).

## 🛠️ Local Setup

### Prerequisites

- Python 3.10+
- uv package manager

### Installation

```bash
# Install uv if not already installed
pip install uv

# Clone the repository
git clone https://github.com/NerdToMars/threed-comp.git
cd threed-comp

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## 📦 Dependencies

All dependencies are managed by `uv` and defined in `pyproject.toml`:

- `ipykernel` - Jupyter kernel support
- `matplotlib` - Plotting and visualization
- `rerun-notebook` - 3D visualization in notebooks
- `rerun-sdk` - 3D visualization SDK

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
