# Driver Drowsiness Detection - Web App

This project provides a small Flask web app that wraps the prediction code exported from the Colab notebook. It allows uploading an image and returns whether the driver is "Drowsy" or "Alert" plus the confidence.

Getting started
1. Export your trained Keras model from the Colab notebook as `model.h5` and place it in the project root (`d:/Drowsiness_Detection/model.h5`).
2. Create a Python virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

3. Run the app:

```powershell
python app.py
```

4. Open http://localhost:5000 in your browser, upload an image, and view results.

Notes
- The predictor expects the same class ordering used in the notebook: `['closed', 'no_yawn', 'open', 'yawn']`.
- The web UI uses a small 128x128 preprocessing to match the notebook's `predict` function.
- If you want to use the Gradio interface from the notebook instead, open the notebook and run the Gradio cell.
