# ¿Quién es y qué siente?

App de Streamlit del **TP1 de IA y Aprendizaje Automático II** (UCA Rosario, 2026) · **Grupo 10**: Franco Do Brito, Juan Cruz Montero y Lucía Formenti.

Se sube una foto de una de las 20 personas del dataset del curso. La app detecta la cara, la recorta igual que en el dataset (cuadrada, con 25 % de margen, 224×224) y predice:

- **Quién es**, con ResNet50 preentrenada en ImageNet (transferencia, con ajuste fino de las últimas 30 capas). En test: 62 % de accuracy sobre 20 personas.
- **Qué emoción muestra** (alegría, sorpresa, enojo, tristeza o neutral), con ResNet50 preentrenada en ImageNet (transferencia, con ajuste fino de las últimas 30 capas). En test: 32 % de accuracy sobre 5 emociones.

Los dos modelos se eligieron por F1 macro en validación, sobre una partición por video (nunca por cuadro) para evitar fugas entre entrenamiento y test.

## Archivos

- `app.py`: la aplicación.
- `modelos/config.json`: la arquitectura, las clases y los resultados de cada modelo.
- `haarcascade_frontalface_default.xml`: el detector de caras de OpenCV.
- Los pesos entrenados (`modelos/*.npz`) no están en el repositorio por su tamaño: la app los descarga de Google Drive la primera vez que arranca.

## Correrla en tu computadora

```bash
pip install -r requirements.txt
streamlit run app.py
```

Funciona con Python 3.10 a 3.12.
