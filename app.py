"""¿Quién es y qué siente? — Grupo 10, IAyAA II (UCA Rosario 2026).

Sube una foto de una de las 20 personas del dataset del curso. La app detecta la cara,
la recorta igual que en el dataset y predice quién es y qué emoción muestra, con los
mejores modelos del notebook (elegidos por F1 macro en validación).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers

CARPETA = Path(__file__).parent / "modelos"
EMOJI = {"alegria": "😄", "sorpresa": "😮", "enojo": "😠", "tristeza": "😢", "neutral": "😐"}

st.set_page_config(page_title="¿Quién es y qué siente?", page_icon="🎭", layout="centered")


# ── Modelos ──────────────────────────────────────────────────────────────────
def construir(cfg):
    """Arma la misma arquitectura que en el notebook y le carga los pesos entrenados."""
    tam, n = cfg["tam"], len(cfg["clases"])
    bases = {"MobileNetV2": tf.keras.applications.MobileNetV2,
             "EfficientNetB0": tf.keras.applications.EfficientNetB0,
             "ResNet50": tf.keras.applications.ResNet50}
    base = bases[cfg["base"]](input_shape=(tam, tam, 3), include_top=False, weights=None)
    ent = tf.keras.Input((tam, tam, 3))
    if cfg["base"] == "MobileNetV2":
        x = layers.Rescaling(1 / 127.5, offset=-1)(ent)                     # [0, 255] → [−1, 1]
    elif cfg["base"] == "ResNet50":
        x = layers.Lambda(tf.keras.applications.resnet50.preprocess_input)(ent)
    else:                                                                     # EfficientNetB0 escala adentro
        x = ent
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(cfg["dropout"])(x)
    modelo = tf.keras.Model(ent, layers.Dense(n, activation="softmax")(x))

    pesos = CARPETA / cfg["pesos"]
    if not pesos.exists() and cfg.get("drive_id"):                            # pesos grandes: se bajan del Drive
        import gdown
        gdown.download(id=cfg["drive_id"], output=str(pesos), quiet=True)
    with np.load(pesos) as z:
        modelo.set_weights([z[f"arr_{i}"] for i in range(len(z.files))])
    return modelo


@st.cache_resource(show_spinner="Cargando los modelos…")
def cargar():
    config = json.loads((CARPETA / "config.json").read_text(encoding="utf-8"))
    return config, {tarea: construir(cfg) for tarea, cfg in config.items()}


# El detector de caras viaja con la app (el archivo .xml): así no depende de dónde lo instale cada versión de OpenCV.
DETECTOR = cv2.CascadeClassifier(str(Path(__file__).parent / "haarcascade_frontalface_default.xml"))


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
st.title("🎭 ¿Quién es y qué siente?")
st.caption("TP1 · IAyAA II · UCA Rosario 2026 · Grupo 10: Franco Do Brito, Juan Cruz Montero, Lucía Formenti")

config, modelos = cargar()

with st.expander("¿Cómo funciona?"):
    personas = ", ".join(bonito(c) for c in config["identidad"]["clases"])
    st.markdown(
        f"Subí una foto de alguna de estas **{len(config['identidad']['clases'])} personas**: {personas}.\n\n"
        "La app busca la cara, la recorta igual que en el dataset (cuadrada, con margen, 224×224) y dos redes "
        "entrenadas por transferencia predicen **quién es** y **qué emoción muestra** "
        f"(identidad: {config['identidad']['base']}; emoción: {config['emocion']['base']}).\n\n"
        "Funciona mejor con la cara de frente, bien iluminada y sola en la foto. Si la persona no es una de las "
        "de la lista, igual va a contestar con el nombre más parecido: el modelo no sabe decir \"no la conozco\"."
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
x = rgb.astype("float32")[None]                                              # mismo formato que en el notebook

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

if "accuracy_test" in config["identidad"]:
    st.caption(
        f"Para tener en cuenta: en el test (videos que el modelo nunca vio) acertó "
        f"{config['identidad']['accuracy_test']:.0%} en identidad y {config['emocion']['accuracy_test']:.0%} en emoción."
    )
