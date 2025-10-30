import os
import numpy as np
from PIL import Image

# Lazy-load model to avoid heavy imports at import time
_model = None
_class_names = ['closed', 'no_yawn', 'open', 'yawn']


def _ensure_model(model_path=None):
    """Ensure the model is loaded. If not loaded, try to load from model_path or './model.h5'.

    Returns the loaded model.
    Raises FileNotFoundError if no model file is found.
    """
    global _model
    if _model is not None:
        return _model

    # lazy import TensorFlow to keep import-light
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError as e:
        raise RuntimeError(f"TensorFlow is required for prediction. Install tensorflow in your environment. Error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error importing TensorFlow: {e}") from e

    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'model.h5')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please export your trained model as 'model.h5' and place it in the project folder.")

    # load the model without compiling to avoid dependency on optimizer state
    _model = tf.keras.models.load_model(model_path, compile=False)
    return _model


def _preprocess_image(img, target_size=(224, 224)):
    """Accept a PIL Image or numpy array and return a preprocessed numpy array ready for model.predict."""
    if isinstance(img, Image.Image):
        img = img.convert('RGB')
        img = img.resize(target_size)
        arr = np.asarray(img).astype('float32') / 255.0
    else:
        # assume numpy array
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        # resize using PIL to keep dependency small
        arr = Image.fromarray((arr * 255).astype('uint8'))
        arr = arr.resize(target_size)
        arr = np.asarray(arr).astype('float32') / 255.0

    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image_from_pil(pil_image, model_path=None):
    """Predict on a PIL Image. Returns a dict with keys: class_name, confidence, status

    status is 'Drowsy' if class is 'closed' or 'yawn', otherwise 'Alert'.
    """
    model = _ensure_model(model_path=model_path)
    x = _preprocess_image(pil_image, target_size=(224, 224))
    preds = model.predict(x)
    probs = np.asarray(preds[0])
    idx = int(np.argmax(probs))
    class_name = _class_names[idx]
    confidence = float(probs[idx])
    if class_name.lower() in ['closed', 'yawn']:
        status = 'Drowsy'
    else:
        status = 'Alert'

    return {
        'class_name': class_name,
        'confidence': confidence,
        'status': status
    }


def predict_image_from_bytes(image_bytes, model_path=None):
    """Convenience wrapper to accept raw image bytes (e.g., from request.files)."""
    img = Image.open(image_bytes)
    return predict_image_from_pil(img, model_path=model_path)


def is_model_loaded():
    return _model is not None
