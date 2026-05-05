import streamlit as st
import joblib
import numpy as np

# --- Configuración de la página (Diseño Bohemio) ---
st.set_page_config(
    page_title="Predicción Bohemio de Iris 🌸",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS personalizados para el toque bohemio ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Charmonman&family=Open+Sans:wght@300&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Open Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Charmonman', cursive;
        color: #556B2F; /* Olive Drab */
    }
    .stApp {
        background-color: #FFF8DC; /* Cornsilk */
        color: #4A4A4A;
    }
    .stNumberInput > label {
        color: #8B4513; /* Saddle Brown */
        font-weight: bold;
        font-size: 1.1em;
    }
    .stButton > button {
        background-color: #A0522D; /* Sienna */
        color: white;
        border-radius: 15px;
        padding: 10px 20px;
        font-size: 18px;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #CD853F; /* Peru */
        color: #FFFACD; /* Lemon Chiffon */
        transform: scale(1.05);
    }
    .prediction-result {
        background-color: #FFFACD; /* Lemon Chiffon */
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #D2B48C; /* Tan */
        text-align: center;
        margin-top: 20px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-text {
        color: #8B4513; /* Saddle Brown */
        font-size: 26px;
        font-weight: bold;
        font-family: 'Charmonman', cursive;
    }
    </style>
""", unsafe_allow_html=True)

# --- Título y descripción (contenido principal) ---
st.markdown("<h1 style='text-align: center; color: #8B4513;'>🌺 Encanto Floral: Predictor de Iris 🌿</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #556B2F; font-style: italic; font-size: 1.1em;'>Sumérgete en la belleza de las flores Iris y descubre su especie con un toque de magia.</p>", unsafe_allow_html=True)
st.write("""
    ---  
    <div style='text-align: center; color: #6B8E23; font-weight: bold;'>
        Ingresa las medidas de tu flor para desvelar su identidad.
    </div>
    --- 
""", unsafe_allow_html=True)

# --- Cargar Modelo y LabelEncoder ---
try:
    # Asegúrate de que las rutas a tus archivos .joblib sean correctas
    loaded_model = joblib.load('/content/neural_network_model.joblib')
    label_encoder = joblib.load('/content/label_encoder_bien.joblib')
    st.success("🌸 Modelo y codificador cargados exitosamente. ¡Listos para florecer! 🍃")
except FileNotFoundError:
    st.error("❌ Error: Asegúrate de que los archivos 'neural_network_model.joblib' y 'label_encoder_bien.joblib' estén en la ruta `/content/`.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error inesperado al cargar recursos: {e}")
    st.stop()

# --- Entrada de características (en la barra lateral para un diseño más limpio) ---
st.sidebar.markdown("<h2 style='color: #8B4513;'>🌼 Características de la Flor 🌼</h2>", unsafe_allow_html=True)

sepal_length = st.sidebar.number_input(
    "Longitud del Sépalo (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1,
    help="Longitud del sépalo, la parte verde que protege la flor."
)
sepal_width = st.sidebar.number_input(
    "Ancho del Sépalo (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.4,
    step=0.1,
    help="Ancho del sépalo."
)
petal_length = st.sidebar.number_input(
    "Longitud del Pétalo (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1,
    help="Longitud del pétalo, la parte colorida de la flor."
)
petal_width = st.sidebar.number_input(
    "Ancho del Pétalo (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1,
    help="Ancho del pétalo."
)

# --- Botón de predicción ---
st.write("\n\n") # Espacio en blanco
if st.button("✨ ¡Adivina mi Especie Floral! ✨"):
    # Preparar los datos para la predicción
    new_flower_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Realizar la predicción
    prediction_label = loaded_model.predict(new_flower_features)

    # Decodificar la predicción usando el LabelEncoder.classes_
    # Se asume que label_encoder.classes_ contiene los nombres de las especies
    predicted_species = label_encoder.classes_[prediction_label[0]]

    st.markdown("<div class='prediction-result'>", unsafe_allow_html=True)
    st.markdown(f"<p class='prediction-text'>¡Esta flor encantadora es una: 🌿 {predicted_species.capitalize()} 🌿</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Pie de página bohemio ---
st.write("\n\n") # Más espacio
st.markdown("""
    ---  
    <p style='text-align: center; color: #8B4513; font-style: italic;'>
        "Donde las flores danzan al ritmo del viento y el conocimiento florece." 🌻
    </p>
    --- 
""", unsafe_allow_html=True)
