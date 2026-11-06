"""¿Quién es y qué siente? — Grupo 10, IAyAA II (UCA Rosario 2026).

Sube una foto de una de las 20 personas del dataset del curso. La app detecta la cara,
la recorta igual que en el dataset y predice quién es y qué emoción muestra, con los
mejores modelos del notebook (elegidos por F1 macro en validación):

- identidad: FaceNet (red entrenada en caras) + regresión logística;
- emoción: HSEmotion (red entrenada en expresiones faciales) + regresión logística.

Las dos redes no se entrenaron: se usan para sacar la "huella" de cada cara, y encima
va un clasificador chico entrenado con nuestro dataset (modelos/clasificadores.npz).
"""
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import timm
import torch

from facenet_modelo import InceptionResnetV1

CARPETA = Path(__file__).parent / "modelos"
HSE_URL = ("https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/"
           "models/affectnet_emotions/enet_b0_8_best_afew.pt")
EMOJI = {"alegria": "😄", "sorpresa": "😮", "enojo": "😠", "tristeza": "😢", "neutral": "😐"}
TAM = 224                                    # tamaño de los recortes del dataset

st.set_page_config(page_title="¿Quién es y qué siente?", page_icon="🎭", layout="centered")


# ── Modelos ──────────────────────────────────────────────────────────────────
def cargar_hsemotion():
    """EfficientNet-B0 entrenada con AffectNet. Se rearma con timm y se le cargan los pesos publicados."""
    archivo = Path.home() / ".cache" / "hsemotion" / "enet_b0_8_best_afew.pt"
    if not archivo.exists():
        archivo.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HSE_URL, archivo)
    original = torch.load(archivo, weights_only=False, map_location="cpu")
    red = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=8)
    red.load_state_dict({k.replace("classifier.0.", "classifier."): v for k, v in original.state_dict().items()})
    red.classifier = torch.nn.Identity()     # solo la huella de la expresión (1280 números)
    return red.eval()


@st.cache_resource(show_spinner="Cargando los modelos…")
def cargar():
    config = json.loads((CARPETA / "config.json").read_text(encoding="utf-8"))
    with np.load(CARPETA / "clasificadores.npz") as z:
        clasif = {t: (z[f"{t}_W"], z[f"{t}_b"]) for t in ("identidad", "emocion")}
    redes = {"identidad": InceptionResnetV1(pretrained="vggface2").eval(), "emocion": cargar_hsemotion()}
    return config, redes, clasif


MEDIA, DESVIO = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def huella(tarea, rgb, redes):
    """Mismo preprocesado que en el notebook (sección 8 bis y 8 ter)."""
    if tarea == "identidad":                 # FaceNet: 160×160, píxeles a [−1, 1]
        x = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = (x - 127.5) / 128
    else:                                    # HSEmotion: 224×224, normalización de ImageNet
        x = (rgb / 255.0 - MEDIA) / DESVIO
    x = torch.tensor(x.transpose(2, 0, 1)[None]).float()
    with torch.no_grad():
        return redes[tarea](x).numpy()[0]


def probabilidades(h, W, b):
    z = h @ W.T + b                          # regresión logística: softmax de una capa lineal
    e = np.exp(z - z.max())
    return e / e.sum()


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

config, redes, clasif = cargar()

with st.expander("¿Cómo funciona?"):
    personas = ", ".join(bonito(c) for c in config["identidad"]["clases"])
    st.markdown(
        f"Subí una foto de alguna de estas **{len(config['identidad']['clases'])} personas**: {personas}.\n\n"
        "La app busca la cara y la recorta igual que en el dataset (cuadrada, con margen, 224×224). "
        "Después, dos redes ya entrenadas por otros la describen con una \"huella\" de números: "
        "**FaceNet**, entrenada con millones de caras para distinguir personas, y **HSEmotion**, entrenada "
        "con cientos de miles de caras para reconocer expresiones. Sobre esas huellas, dos clasificadores "
        "entrenados con nuestro dataset dicen **quién es** y **qué emoción muestra**.\n\n"
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

cara = recortar_cara(img, TAM)
if cara is None:
    st.warning("No encontré ninguna cara en la foto: uso la imagen entera, y el resultado puede ser peor.")
    cara = cv2.resize(img, (TAM, TAM), interpolation=cv2.INTER_AREA)
rgb = cv2.cvtColor(cara, cv2.COLOR_BGR2RGB)

col_img, col_res = st.columns([1, 1.6])
with col_img:
    st.image(rgb, caption="Cara recortada", width="stretch")

with col_res:
    for tarea, titulo in (("identidad", "¿Quién es?"), ("emocion", "¿Qué emoción muestra?")):
        prob = probabilidades(huella(tarea, rgb, redes), *clasif[tarea])
        clases = config[tarea]["clases"]
        top = prob.argsort()[::-1][:3]
        ganador = clases[top[0]]
        etiqueta = f"{EMOJI.get(ganador, '')} {bonito(ganador)}" if tarea == "emocion" else bonito(ganador)
        st.subheader(titulo)
        st.markdown(f"### {etiqueta}")
        for i in top:
            st.progress(float(prob[i]), text=f"{bonito(clases[i])}: {prob[i]:.0%}")

st.caption(
    f"Para tener en cuenta: en el test (videos que el modelo nunca vio) acertó "
    f"{config['identidad']['accuracy_test']:.0%} en identidad y {config['emocion']['accuracy_test']:.0%} en emoción."
)
