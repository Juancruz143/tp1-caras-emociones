# ¿Quién es y qué siente?

App de Streamlit del **TP1 de IA y Aprendizaje Automático II** (UCA Rosario, 2026) · **Grupo 10**: Franco Do Brito, Juan Cruz Montero y Lucía Formenti.

Se sube una foto de una de las 20 personas del dataset del curso. La app detecta la cara, la recorta igual que en el dataset (cuadrada, con 25 % de margen, 224×224) y predice:

- **Quién es**, con **FaceNet** (red entrenada con millones de caras, dataset VGGFace2) más una regresión logística entrenada con nuestro dataset. En test: 95 % de accuracy sobre 20 personas.
- **Qué emoción muestra** (alegría, sorpresa, enojo, tristeza o neutral), con **HSEmotion** (EfficientNet-B0 entrenada con AffectNet para reconocer expresiones) más una regresión logística entrenada con nuestro dataset. En test: 52 % de accuracy sobre 5 emociones.

Las dos redes no se reentrenan: describen cada cara con una "huella" de números, y el clasificador de encima es lo único que se entrenó con nuestras fotos. Los modelos se eligieron por F1 macro en validación, sobre una partición por video (nunca por cuadro) para evitar fugas entre entrenamiento y test.

## Archivos

- `app.py`: la aplicación.
- `facenet_modelo.py`: la arquitectura de FaceNet, copiada de [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (licencia MIT).
- `modelos/clasificadores.npz`: los dos clasificadores entrenados con nuestro dataset (pesos de la regresión logística).
- `modelos/config.json`: las clases y los resultados de cada modelo.
- `haarcascade_frontalface_default.xml`: el detector de caras de OpenCV.
- Los pesos de FaceNet y HSEmotion se descargan de GitHub la primera vez que arranca la app.

## Correrla en tu computadora

```bash
pip install -r requirements.txt
streamlit run app.py
```

Funciona con Python 3.10 a 3.13.
