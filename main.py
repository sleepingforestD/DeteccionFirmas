import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ====== AQUÍ VAN LAS CONSTANTES DEL MODELO =====================

IMG_HEIGHT = 128   # <-- alto con el que ENTRENASTE (ejemplo)
IMG_WIDTH  = 128   # <-- ancho con el que ENTRENASTE (ejemplo)
CHANNELS   = 1     # <-- 1 si entrenaste en escala de grises, 3 si fue RGB

# ============================================
# Configuración de TensorFlow (GPU opcional)
# ============================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU detectada, memory_growth activado.")
    except Exception as e:
        print(f"No se pudo configurar memory_growth: {e}")

# ============================================
# Inicialización de la API
# ============================================
app = FastAPI(
    title="FalsificacionFirmas",
    description="API para detectar firmas falsas",
    version="1.0.0"
)

# Variable global para el modelo
model = None

# ============================================
# Carga del modelo
# ============================================
def load_model():
    """Cargar el modelo entrenado desde FalsificacionFirmas.h5"""
    global model
    model_path = "FalsificacionFirmas.h5"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
    
    try:
        model = keras.models.load_model(model_path)
        print("Modelo cargado exitosamente.")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise

# ============================================
# Preprocesamiento de imagen
# ============================================
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesar imagen siguiendo los mismos pasos del entrenamiento:
    1. Redimensionar a IMG_WIDTH x IMG_HEIGHT
    2. Convertir a RGB o escala de grises según CHANNELS
    3. Normalizar a [0, 1]
    4. Añadir dimensión de batch
    """
    try:
        # 1. Escala de grises (equivalente a cv2.IMREAD_GRAYSCALE)
        image = image.convert("L")

        # 2. Redimensionar al tamaño usado en el entrenamiento
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))  # (128, 128)

        # 3. Convertir a array numpy
        img_array = np.array(image)  # shape: (128, 128)

        # 4. Añadir canal (H, W, 1)
        if img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=-1)

        # 5. Normalizar como en el notebook: / 255.0
        img_array = img_array.astype("float32") / 255.0

        # 6. Añadir dimensión batch: (1, 128, 128, 1)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar la imagen: {str(e)}"
        )

# ============================================
# Evento de inicio: cargar modelo al arrancar
# ============================================
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"Error al iniciar la aplicación: {e}")

# ============================================
# Endpoints
# ============================================
@app.get("/")
async def root():
    return {
        "message": "Firmas reales vs falsas API",
        "status": "active",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict  (POST, imagen)",
            "health": "/health   (GET, estado del modelo)"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tensorflow_version": tf.__version__
    }

@app.post("/predict")
async def predict_imagen(file: UploadFile = File(...)):
    """
    Predecir si la firma es falsa o real.

    Parámetros:
        file: imagen enviada vía multipart/form-data.

    Respuesta:
        - class: 0 para firma falsa, 1 para firma real
        - class_name: "falsa" o "real"
        - confidence: probabilidad de la clase predicha
        - probabilities: probabilidad para cada clase
        - image_info: metadatos básicos de la imagen
    """
    # Verificar que el modelo esté cargado
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo no cargado. Revisar logs del servidor."
        )

    # Verificar tipo de archivo
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (content-type 'image/*')."
        )

    try:
        # Leer la imagen en memoria
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocesar la imagen
        processed_image = preprocess_image(image)

        # Predicción
        prediction = model.predict(processed_image)

        # Extraer resultados
        probabilities = prediction[0].tolist()
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        # Mapeo de clases
        class_names = {0: "falsa", 1: "real"}
        class_name = class_names.get(predicted_class, "desconocida")

        return {
            "class": predicted_class,
            "class_name": class_name,
            "confidence": confidence,
            "probabilities": {
                "falsa": probabilities[0],
                "real": probabilities[1]
            },
            "image_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "processed_shape": processed_image.shape
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

# ============================================
# Entrada principal (para ejecutar con python main.py)
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



