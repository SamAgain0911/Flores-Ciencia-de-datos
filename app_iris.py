import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="Predicción de Iris 🌸",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

def inject_css():
    css_lines = [
        "<style>",
        "@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400&display=swap');",
        "html, body, [class*='st-'] { font-family: 'Lato', sans-serif; }",
        "h1, h2, h3 { font-family: 'Playfair Display', serif; }",
        ".stApp { background: linear-gradient(135deg, #fdf6ec 0%, #f5ede0 100%); color: #3d2b1f; }",
        "[data-testid='stSidebar'] { background: linear-gradient(180deg, #3d1f0f 0%, #6b3a2a 100%); }",
        "[data-testid='stSidebar'] * { color: #f5ede0 !important; }",
        "[data-testid='stSidebar'] label { font-weight: 600 !important; }",
        "[data-testid='stMetric'] { background-color: rgba(139,69,19,0.08); border: 1px solid rgba(139,69,19,0.2); border-radius: 12px; padding: 10px 12px; }",
        "[data-testid='stMetricLabel'] p { color: #6B4226 !important; font-weight: 600; }",
        "[data-testid='stMetricValue'] { color: #8B4513 !important; font-weight: 700; }",
        ".stButton > button { background: linear-gradient(135deg, #8B4513, #c0692a); color: white; border-radius: 30px; padding: 12px 30px; font-size: 17px; border: none; width: 100%; box-shadow: 0 4px 15px rgba(139,69,19,0.4); transition: all 0.3s ease; }",
        ".stButton > button:hover { background: linear-gradient(135deg, #c0692a, #e08030); transform: translateY(-2px); }",
        ".prediction-card { background: linear-gradient(135deg, #fffaf0, #fff5e1); padding: 30px 25px; border-radius: 20px; border: 2px solid #D2B48C; text-align: center; margin-top: 25px; box-shadow: 0 8px 25px rgba(139,69,19,0.15); }",
        ".prediction-species { color: #8B4513; font-size: 32px; font-family: 'Playfair Display', serif; font-weight: 700; font-style: italic; margin: 5px 0; }",
        ".prediction-common { color: #a07850; font-size: 15px; margin: 0 0 12px 0; }",
        ".chip { display: inline-block; background-color: rgba(139,69,19,0.1); border: 1px solid rgba(139,69,19,0.3); border-radius: 20px; padding: 5px 14px; font-size: 13px; color: #6B4226; margin: 4px; }",
        "#MainMenu { visibility: hidden; }",
        "footer { visibility: hidden; }",
        "</style>"
    ]
    st.markdown("\n".join(css_lines), unsafe_allow_html=True)

inject_css()

st.markdown(
    "<div style='text-align:center; padding:20px 0 5px 0;'>"
    "<span style='font-size:48px;'>🌺</span>"
    "<h1 style='color:#6B3A2A; margin:5px 0;'>Predictor de Iris</h1>"
    "<p style='color:#9b6a4a; font-style:italic;'>Descubre la especie de tu flor con inteligencia artificial</p>"
    "</div>"
    "<p style='text-align:center; color:#c0a080; letter-spacing:8px;'>❧ · · · ❧</p>",
    unsafe_allow_html=True
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "neural_network_model.joblib"))
    return model

try:
    loaded_model = load_model()
    st.success("✅ Modelo cargado correctamente · Listo para predecir")
except FileNotFoundError as e:
    st.error("❌ Archivo no encontrado: " + str(e))
    st.stop()
except Exception as e:
    st.error("❌ Error inesperado: " + str(e))
    st.stop()

# Índice 0 = setosa, 1 = versicolor, 2 = virginica (orden del dataset UCI Iris)
SPECIES_INFO = {
    0: {"scientific": "Iris setosa",     "common": "Lirio cerda · Flores pequeñas, sépalos anchos", "emoji": "🌸"},
    1: {"scientific": "Iris versicolor", "common": "Lirio bandera azul · Tamaño intermedio",         "emoji": "🌺"},
    2: {"scientific": "Iris virginica",  "common": "Lirio virginiano · La especie más grande",        "emoji": "🌷"},
}

st.sidebar.markdown(
    "<div style='text-align:center; padding:10px 0 20px 0;'>"
    "<span style='font-size:36px;'>🌼</span>"
    "<h2 style='font-size:1.4em; margin:5px 0;'>Medidas de la Flor</h2>"
    "<p style='font-size:0.82em; opacity:0.75;'>Valores en centímetros</p>"
    "</div>",
    unsafe_allow_html=True
)

st.sidebar.markdown("**🟢 Sépalo**")
sepal_length = st.sidebar.number_input("Longitud del Sépalo (cm)", 0.0, 10.0, 5.1, 0.1)
sepal_width  = st.sidebar.number_input("Ancho del Sépalo (cm)",    0.0, 10.0, 3.5, 0.1)
st.sidebar.markdown("**🌸 Pétalo**")
petal_length = st.sidebar.number_input("Longitud del Pétalo (cm)", 0.0, 10.0, 1.4, 0.1)
petal_width  = st.sidebar.number_input("Ancho del Pétalo (cm)",    0.0, 10.0, 0.2, 0.1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sépalo largo", f"{sepal_length} cm")
col2.metric("Sépalo ancho", f"{sepal_width} cm")
col3.metric("Pétalo largo", f"{petal_length} cm")
col4.metric("Pétalo ancho", f"{petal_width} cm")

if st.button("🔮  Predecir Especie Floral"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_idx = int(loaded_model.predict(features)[0])
    info     = SPECIES_INFO.get(pred_idx, {"scientific": "Desconocida", "common": "Índice: " + str(pred_idx), "emoji": "🌿"})

    st.markdown(
        "<div class='prediction-card'>"
        "<span style='font-size:52px;'>" + info["emoji"] + "</span>"
        "<p style='color:#6B4226; font-size:13px; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:4px;'>Especie identificada</p>"
        "<p class='prediction-species'>" + info["scientific"] + "</p>"
        "<p class='prediction-common'>" + info["common"] + "</p>"
        "<span class='chip'>📏 Sépalo " + str(sepal_length) + " × " + str(sepal_width) + " cm</span>"
        "<span class='chip'>🌿 Pétalo " + str(petal_length) + " × " + str(petal_width) + " cm</span>"
        "</div>",
        unsafe_allow_html=True
    )

st.markdown(
    "<p style='text-align:center; color:#c0a080; letter-spacing:8px; margin-top:40px;'>❧ · · · ❧</p>"
    "<p style='text-align:center; color:#a07850; font-style:italic; font-size:0.9em;'>"
    "\"La naturaleza siempre lleva los colores del espíritu.\" — Emerson 🌻</p>",
    unsafe_allow_html=True
)
