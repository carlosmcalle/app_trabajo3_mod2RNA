import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import plotly.express as px

# --- CONFIGURACIÓN ---
class_names = ['other_activities', 'safe_driving', 'talking_phone', 'texting_phone', 'turning']

@st.cache_resource
def load_model():
    model_path = Path(__file__).parents[1] / "models" / "model_module_2.keras"
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((64, 64))
    image_array = np.array(image)  # ¡NO dividir por 255!
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# --- INTERFAZ ---
st.title("🚗 Clasificación de Actividades del Conductor")
st.markdown("Este módulo detecta si el conductor está realizando alguna actividad riesgosa.")

model = load_model()
if model is None:
    st.stop()

# Subida de imagen
uploaded_file = st.file_uploader("📷 Sube una imagen del conductor", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Imagen cargada", use_container_width=True)

    # Procesamiento
    processed_img = preprocess_image(image)

    # Predicción
    predictions = model.predict(processed_img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    class_pred = {
        "other_activities": "Otras actividades",
        "safe_driving": "Conducción segura",
        "talking_phone": "Hablando por teléfono",
        "texting_phone": "Escribiendo en el celular",
        "turning": "Girando"
    }.get(predicted_class)

    st.subheader("📊 Resultado de Clasificación")
    st.write(f"**Actividad Detectada:** `{class_pred}`")
    st.write(f"**Confianza del modelo:** `{confidence:.2f}%`")

    if predicted_class == "talking_phone":
        st.error("📱⚠️ El conductor está hablando por teléfono. Esto representa una distracción peligrosa.")
    elif predicted_class == "texting_phone":
        st.error("💬⚠️ El conductor está escribiendo en el celular. Alta probabilidad de accidente por distracción visual y manual.")
    elif predicted_class == "safe_driving":
        st.success("✅ Conducción segura detectada. El conductor está enfocado en la vía.")
    elif predicted_class == "other_activities":
        st.warning("🔍 El conductor está realizando otra actividad que podría implicar distracción. Se recomienda monitorear.")
    elif predicted_class == "turning":
        st.info("↩️ El conductor está realizando un giro. Aunque es una acción esperada, requiere atención en maniobras.")
    else:
        st.info("ℹ️ Actividad desconocida o no clasificada.")

    with st.expander("Ver probabilidades por clase"):
        prob_values = predictions[0]
        prob_data = {
            "Actividad": class_names,
            "Probabilidad": prob_values
        }
        
        fig = px.bar(
            prob_data,
            x="Probabilidad",
            y="Actividad",
            orientation='h',
            text=[f"{p:.2%}" for p in prob_values],
            labels={"Actividad": "Clase", "Probabilidad": "Probabilidad"},
            title="Distribución de Probabilidades por Clase"
        )
        fig.update_traces(marker_color='steelblue', textposition='outside')
        fig.update_layout(xaxis_range=[0, 1], yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
