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

## Versión con la CNN desde cero (`app_cnn/`)

Una segunda app, aparte, que usa la **CNN desde cero** del notebook (Conv 32 → Conv 64 → Flatten → Densa 128), una para identidad y otra para emoción, sin ninguna red preentrenada. Sirve para comparar con la app principal.

- Se reentrenó con la misma arquitectura, la misma partición 80/15/5 por video, la misma aumentación, los mismos pesos por clase y los mismos callbacks que el notebook (en CPU).
- En test: 34 % en identidad y 29 % en emoción (validación: 58 % y 26 %). La CNN desde cero es inestable: cada entrenamiento da números algo distintos, y en el notebook había dado 42 % y 19 % en test.
- Los pesos van en `app_cnn/cnn_*.npz`, guardados en float16 para que cada archivo pese menos de 50 MB.
- Para publicarla en Streamlit: mismo repositorio, rama `main`, archivo principal `app_cnn/app.py` (usa su propio `app_cnn/requirements.txt`, con TensorFlow).

## Presentación

La presentación del TP está en [`docs/TP1_Entrega2_Modelos.pptx`](docs/TP1_Entrega2_Modelos.pptx) ([descarga directa](https://github.com/Juancruz143/tp1-caras-emociones/raw/main/docs/TP1_Entrega2_Modelos.pptx)).
