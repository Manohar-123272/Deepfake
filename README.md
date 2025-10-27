# Deepfake

This repository provides code for deepfake detection using advanced transformer architectures, such as Swin Transformer and DIET Transformer, along with Grad-CAM visualizations for model explainability.

Folder Structure
dataset/ — Place your dataset here.

best_SwinTransformer_model.pth — Pretrained Swin Transformer model weights.

diet_transformer_fixed.py — DIET Transformer implementation (fixed version).

swin_transformer_fixed.py — Swin Transformer implementation (fixed version).

main.py — Main script to run training, testing, and Grad-CAM visualizations.

requirements.txt — Python dependencies for this project.

Getting Started
1. Clone the Repository
bash
git clone [https://github.com/yourusername/DEEPFAKE.git](https://github.com/Manohar-123272/Deepfake)
cd DEEPFAKE
2. Install Requirements
It is recommended to use a virtual environment.
bash
pip install -r requirements.txt



4. Prepare the Dataset
Place your deepfake dataset inside the dataset/ folder.

Ensure the directory and file structure matches the expected format in main.py.

4. Run the Script
To train or evaluate the deepfake detection model and generate Grad-CAM visualizations:

bash
python main.py
You may need to edit main.py if you want to change hyperparameters, model selection, or visualization settings.

5. Pre-trained Model Usage
To use the provided best Swin Transformer checkpoint, ensure that best_SwinTransformer_model.pth is in the root directory and the code in main.py loads this checkpoint.

Features
Transformer-based models for accurate deepfake detection.

Grad-CAM visualizations to interpret model predictions and highlight manipulated regions.

Notes:
Scripts are intended for research and educational purposes.

Make sure your dataset is properly preprocessed as per the model requirements.

Edit scripts as necessary for your custom use-cases.

Acknowledgments
This project uses open-source implementations of Swin Transformer and DIET Transformer.

License
This project is licensed under the MIT License - see the LICENSE file for details.

MIT License
MIT License

Copyright (c) 2025 [Manohar]

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
