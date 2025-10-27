# 🎭 Deepfake Detection with Transformer Models

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Advanced deepfake detection system** using state-of-the-art transformer architectures with explainable AI through Grad-CAM visualizations.

---

## 🌟 Features

- ✅ **Transformer-based models** for accurate deepfake detection
- 🔍 **Grad-CAM visualizations** to interpret model predictions and highlight manipulated regions
- 🚀 **Pre-trained model weights** included for quick deployment
- 📊 **Support for Swin Transformer & DIET Transformer** architectures

---

## 📁 Folder Structure

```
DEEPFAKE/
├── dataset/                              # Place your dataset here
├── best_SwinTransformer_model.pth       # Pre-trained Swin Transformer weights
├── diet_transformer_fixed.py            # DIET Transformer implementation
├── swin_transformer_fixed.py            # Swin Transformer implementation
├── main.py                              # Main training & evaluation script
└── requirements.txt                      # Python dependencies
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Manohar-123272/Deepfake.git
cd Deepfake
```

### 2️⃣ Install Requirements

**It is recommended to use a virtual environment.**

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Prepare the Dataset

- 📂 Place your deepfake dataset inside the `dataset/` folder
- ⚙️ Ensure the directory structure matches the expected format in `main.py`

### 4️⃣ Run the Script

**To train or evaluate the model and generate Grad-CAM visualizations:**

```bash
python main.py
```

> 💡 **Tip**: Edit `main.py` to customize hyperparameters, model selection, or visualization settings.

### 5️⃣ Pre-trained Model Usage

- Ensure `best_SwinTransformer_model.pth` is in the root directory
- The model checkpoint will be loaded automatically by `main.py`

---

## 🎯 Key Highlights

| Feature | Description |
|---------|-------------|
| **🧠 Advanced Architectures** | Swin Transformer & DIET Transformer for state-of-the-art detection |
| **👁️ Explainable AI** | Grad-CAM visualizations show which regions triggered detection |
| **⚡ Ready-to-Use** | Pre-trained weights included for immediate deployment |
| **🔬 Research-Ready** | Perfect for academic research and experimentation |

---

## 📝 Notes

- 🎓 **Scripts are intended for research and educational purposes**
- 🔧 Make sure your dataset is properly preprocessed as per model requirements
- ✏️ Edit scripts as necessary for your custom use-cases
- 📊 Supports various deepfake datasets (DeepFake Detection Challenge, FaceForensics++, etc.)

---

## 🙏 Acknowledgments

This project uses open-source implementations of:
- **Swin Transformer** - Hierarchical Vision Transformer
- **DIET Transformer** - Dual-Intent Entity Transformer

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

```
Copyright (c) 2025 Manohar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📬 Contact & Support

For questions, issues, or contributions:
- 🐛 [Report Issues](https://github.com/Manohar-123272/Deepfake/issues)
- 💬 [Discussions](https://github.com/Manohar-123272/Deepfake/discussions)
- ⭐ **Star this repo** if you find it helpful!

---

<div align="center">

**Made with ❤️ for AI Research Community**

</div>
