"""¿Quién es y qué siente? — versión con la CNN desde cero. Grupo 10, IAyAA II (UCA Rosario 2026).

Es la misma app que la principal, pero los dos modelos (identidad y emoción) son la CNN que
armamos y entrenamos desde cero con nuestro dataset, sin ninguna red preentrenada:
Conv(32) → MaxPool → Conv(64) → MaxPool → Dropout → Flatten → Densa(128) → Dropout → salida.

Sirve para comparar: muestra qué tan lejos queda una red desde cero, con ~2000 fotos,
de los modelos preentrenados de la app principal.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers

CARPETA = Path(__file__).parent
EMOJI = {"alegria": "😄", "sorpresa": "😮", "enojo": "😠", "tristeza": "😢", "neutral": "😐"}

st.set_page_config(page_title="¿Quién es y qué siente? · CNN", page_icon="🧱", layout="centered")


# ── Modelos ──────────────────────────────────────────────────────────────────
def cnn_desde_cero(n, tam):
    """La misma arquitectura que la sección 6 del notebook."""
    return tf.keras.Sequential([
        tf.keras.Input((tam, tam, 3)),
        layers.Rescaling(1 / 255),
        layers.Conv2D(32, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Dropout(0.25), layers.Flatten(),
        layers.Dense(128, activation="relu"), layers.Dropout(0.5),
        layers.Dense(n, activation="softmax")])


@st.cache_resource(show_spinner="Cargando las CNN…")
def cargar():
    config = json.loads((CARPETA / "config_cnn.json").read_text(encoding="utf-8"))
    modelos = {}
    for tarea, cfg in config.items():
        m = cnn_desde_cero(len(cfg["clases"]), cfg["tam"])
        with np.load(CARPETA / cfg["pesos"]) as z:          # pesos guardados en float16 para que el archivo sea chico
            m.set_weights([z[f"arr_{i}"].astype(np.float32) for i in range(len(z.files))])
        modelos[tarea] = m
    return config, modelos


# El detector de caras viaja con la app (el archivo .xml): así no depende de dónde lo instale cada versión de OpenCV.
DETECTOR = cv2.CascadeClassifier(str(CARPETA / "haarcascade_frontalface_default.xml"))


def recortar_cara(img, tam, margen=0.25):
    """Busca la cara más grande y la recorta cuadrada con margen, como en el dataset."""
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    caras = DETECTOR.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(caras) == 0:
        return None
    x, y, w, h = max(caras, key=lambda c: c[2] * c[3])
    lado = int(max(w, h) * (1 + 2 * margen))
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(cx - lado // 2, 0), max(cy - lado // 2, 0)
    x1, y1 = min(x0 + lado, img.shape[1]), min(y0 + lado, img.shape[0])
    return cv2.resize(img[y0:y1, x0:x1], (tam, tam), interpolation=cv2.INTER_AREA)


def bonito(nombre):
    return nombre.replace("_", " ").title()


# ── Interfaz ─────────────────────────────────────────────────────────────────
st.title("🧱 ¿Quién es y qué siente? · CNN desde cero")
st.caption("TP1 · IAyAA II · UCA Rosario 2026 · Grupo 10: Franco Do Brito, Juan Cruz Montero, Lucía Formenti")

config, modelos = cargar()

with st.expander("¿Cómo funciona?"):
    personas = ", ".join(bonito(c) for c in config["identidad"]["clases"])
    st.markdown(
        f"Subí una foto de alguna de estas **{len(config['identidad']['clases'])} personas**: {personas}.\n\n"
        "Esta versión usa **la CNN que armamos desde cero**, sin ninguna red preentrenada: dos capas "
        "convolucionales, una capa densa y la salida, entrenadas solo con las fotos de train de nuestro dataset "
        "(una red para identidad y otra para emoción).\n\n"
        "Está para **comparar**: con ~25 millones de parámetros y menos de 2000 fotos, la red memoriza más de lo "
        "que aprende, así que se equivoca mucho más que la app principal (FaceNet y HSEmotion)."
    )

archivo = st.file_uploader("Subí una foto", type=["jpg", "jpeg", "png", "webp"])
if archivo is None:
    st.info("Esperando una foto…")
    st.stop()

img = cv2.imdecode(np.frombuffer(archivo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if img is None:
    st.error("No pude abrir la imagen. Probá con otro archivo (JPG o PNG).")
    st.stop()

tam = config["identidad"]["tam"]
cara = recortar_cara(img, tam)
if cara is None:
    st.warning("No encontré ninguna cara en la foto: uso la imagen entera, y el resultado puede ser peor.")
    cara = cv2.resize(img, (tam, tam), interpolation=cv2.INTER_AREA)
rgb = cv2.cvtColor(cara, cv2.COLOR_BGR2RGB)
x = rgb.astype("float32")[None]                     # mismo formato que en el notebook: 0–255 (la red escala adentro)

col_img, col_res = st.columns([1, 1.6])
with col_img:
    st.image(rgb, caption="Cara recortada", width="stretch")

with col_res:
    for tarea, titulo in (("identidad", "¿Quién es?"), ("emocion", "¿Qué emoción muestra?")):
        prob = modelos[tarea].predict(x, verbose=0)[0]
        clases = config[tarea]["clases"]
        top = prob.argsort()[::-1][:3]
        ganador = clases[top[0]]
        etiqueta = f"{EMOJI.get(ganador, '')} {bonito(ganador)}" if tarea == "emocion" else bonito(ganador)
        st.subheader(titulo)
        st.markdown(f"### {etiqueta}")
        for i in top:
            st.progress(float(prob[i]), text=f"{bonito(clases[i])}: {prob[i]:.0%}")

st.caption(
    f"Para tener en cuenta: en el test (videos que el modelo nunca vio) esta CNN acertó "
    f"{config['identidad']['accuracy_test']:.0%} en identidad y {config['emocion']['accuracy_test']:.0%} en emoción. "
    "La app principal, con FaceNet y HSEmotion, acierta 95 % y 52 %."
)
