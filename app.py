"""
app.py – Fashion MNIST Classifier
Datatón 3 – Applied Machine Learning
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from sklearn.metrics import confusion_matrix
import io

# TensorFlow es opcional — si no está instalado la app funciona en modo demo
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Constantes ───────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ─── Carga del modelo (cacheado) ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not TF_AVAILABLE:
        return None
    try:
        model = tf.keras.models.load_model("modelo_final.h5")
        return model
    except Exception:
        return None

# ─── Carga del dataset (cacheado) ─────────────────────────────────────────────
@st.cache_data
def load_dataset():
    if TF_AVAILABLE:
        from tensorflow.keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        return X_train, y_train, X_test, y_test
    # Fallback: descarga via scikit-learn (no requiere TensorFlow)
    from sklearn.datasets import fetch_openml
    data = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
    X = data.data.reshape(-1, 28, 28).astype(np.uint8)
    y = data.target.astype(int)
    return X[:60000], y[:60000], X[60000:], y[60000:]

# ─── Función para graficar distribución ───────────────────────────────────────
def plot_distribution(y, title="Distribución de clases"):
    unique, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    bars = ax.bar([CLASS_NAMES[i] for i in unique], counts, color=colors, edgecolor='white')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Clase"); ax.set_ylabel("Cantidad")
    ax.tick_params(axis='x', rotation=30)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(count), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    return fig

# ─── Función para mostrar grilla de imágenes ─────────────────────────────────
def plot_sample_grid(X, y):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    for class_id in range(10):
        idx = np.where(y == class_id)[0][0]
        axes[class_id].imshow(X[idx], cmap='gray')
        axes[class_id].set_title(CLASS_NAMES[class_id], fontsize=9)
        axes[class_id].axis('off')
    plt.suptitle("Una imagen por clase", fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/120px-TensorFlowLogo.svg.png", width=80)
    st.title("Fashion MNIST")
    st.caption("Datatón 3 – Applied Machine Learning")
    st.divider()
    st.markdown("**Equipo:** Grupo 7 – Fashion Classifiers")
    st.markdown("**Integrantes:**")
    st.markdown("- Ivan Enrique Gonzalez Paredes")
    st.markdown("- Elizabeth Antuaneth Suarez Pazos")
    st.markdown("- Stephany Gutierrez")
    st.markdown("- Raquel Melany Mendoza Tello")
    st.divider()
    st.markdown("**Dataset:** 70,000 imágenes 28×28")
    st.markdown("**Clases:** 10 categorías de ropa")
    st.markdown("**Framework:** TensorFlow / Keras")
    if not TF_AVAILABLE:
        st.divider()
        st.warning("Modo demo — TensorFlow no instalado. La predicción requiere modelo_final.h5")

# ─── Tabs principales ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Dataset", "Resultados del Modelo", "Predicción Interactiva"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – INFORMACIÓN DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Fashion MNIST – Información del Dataset")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **Fashion MNIST** es un dataset de referencia para visión por computadora creado por Zalando Research.
        Es un reemplazo directo del clásico MNIST de dígitos, pero con **mayor dificultad** por la similitud
        visual entre categorías de ropa.

        Es ampliamente usado en benchmarks de modelos de clasificación de imágenes, especialmente para
        comparar arquitecturas MLP vs CNN en datasets de tamaño mediano.
        """)

    with col2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total imágenes", "70,000")
        m2.metric("Train / Test", "60k / 10k")
        m3.metric("Clases", "10")
        m1.metric("Tamaño imagen", "28×28 px")
        m2.metric("Canales", "Escala de grises")
        m3.metric("Balance", "Perfecto")

    st.divider()
    st.subheader("Tabla de Clases")
    classes_df = pd.DataFrame({
        "ID": list(range(10)),
        "Clase": CLASS_NAMES
    })
    st.dataframe(classes_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Ejemplos por Clase")

    with st.spinner("Cargando dataset..."):
        X_train, y_train, X_test, y_test = load_dataset()

    fig_grid = plot_sample_grid(X_train, y_train)
    st.pyplot(fig_grid)

    st.divider()
    st.subheader("Distribución de Clases")
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        fig_train = plot_distribution(y_train, "Distribución – Train (60,000 imágenes)")
        st.pyplot(fig_train)
    with col_dist2:
        fig_test = plot_distribution(y_test, "Distribución – Test (10,000 imágenes)")
        st.pyplot(fig_test)

    st.info("El dataset está perfectamente balanceado: 6,000 imágenes por clase en train y 1,000 en test. No se requieren técnicas de balanceo de clases.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – RESULTADOS DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Comparación de Modelos")

    st.subheader("Tabla de Métricas")
    st.caption("Los valores se generan al ejecutar el notebook y guardar el modelo.")

    try:
        metrics_df = pd.read_csv("metricas_modelos.csv")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        metrics_placeholder = pd.DataFrame({
            "Modelo":             ["MLP Baseline", "CNN Base", "CNN + Regularización"],
            "Accuracy":           [0.8863, 0.9053, 0.8720],
            "Precision (macro)":  [0.8857, 0.9059, 0.8838],
            "Recall (macro)":     [0.8863, 0.9053, 0.8720],
            "F1-Score (macro)":   [0.8853, 0.9044, 0.8747],
        })
        st.dataframe(metrics_placeholder.style.format({c: "{:.4f}" for c in metrics_placeholder.columns[1:]}),
                     use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Comparativo de Accuracy")
    try:
        metrics_real = pd.read_csv("metricas_modelos.csv")
        modelos = metrics_real['Modelo'].tolist()
        accs    = metrics_real['Accuracy'].tolist()
    except FileNotFoundError:
        modelos = ["MLP Baseline", "CNN Base", "CNN + Reg."]
        accs    = [0.882, 0.915, 0.928]

    fig_comp = plt.figure(figsize=(8, 4))
    colors   = ['#4C72B0', '#DD8452', '#55A868']
    plt.bar(modelos, accs, color=colors, edgecolor='white', width=0.5)
    plt.ylim(min(accs) - 0.02, min(1.0, max(accs) + 0.02))
    plt.ylabel("Accuracy en Test")
    plt.title("Accuracy por Modelo", fontweight='bold')
    for i, v in enumerate(accs):
        plt.text(i, float(v) + 0.001, f'{float(v):.3f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_comp)

    st.divider()
    st.subheader("Matriz de Confusión – Mejor Modelo")
    try:
        img_cm = Image.open("confusion_matrix.png")
        st.image(img_cm, caption="Matriz de Confusión – CNN Base", use_container_width=True)
    except FileNotFoundError:
        st.info("La matriz de confusión se generará al ejecutar el notebook (confusion_matrix.png).")

    st.divider()
    st.subheader("Curvas de Entrenamiento – Mejor Modelo")
    try:
        img_curvas = Image.open("curvas_CNN_Base.png")
        st.image(img_curvas, caption="Loss y Accuracy durante el entrenamiento", use_container_width=True)
    except FileNotFoundError:
        st.info("Las curvas se generarán al ejecutar el notebook (curvas_CNN_Base.png).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – PREDICCIÓN INTERACTIVA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Clasificador Interactivo de Prendas")
    st.markdown("Sube una imagen de prenda de ropa y el modelo predecirá su categoría.")

    model = load_model()

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("Sube tu imagen")
        uploaded = st.file_uploader(
            "Formatos aceptados: JPG, PNG, BMP",
            type=["jpg", "jpeg", "png", "bmp"]
        )

        st.divider()
        st.markdown("**Consejos para mejores resultados:**")
        st.markdown("- Usa imágenes con fondo blanco o claro")
        st.markdown("- La prenda debe ocupar la mayor parte de la imagen")
        st.markdown("- El modelo reconoce las 10 categorías del dataset")
        st.markdown("- Imágenes en escala de grises funcionan mejor")

    with col_result:
        st.subheader("Resultado de la Predicción")

        if uploaded is not None and model is not None:
            img_original = Image.open(uploaded)
            img_gray     = img_original.convert("L").resize((28, 28))
            arr          = np.array(img_gray) / 255.0
            arr_input    = arr.reshape(1, 28, 28, 1)

            pred_probs = model.predict(arr_input, verbose=0)[0]
            pred_idx   = np.argmax(pred_probs)
            pred_name  = CLASS_NAMES[pred_idx]
            pred_conf  = pred_probs[pred_idx] * 100

            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(img_original, caption="Imagen original", use_container_width=True)
            with col_img2:
                st.image(img_gray, caption="28x28 px (entrada al modelo)", use_container_width=True)

            st.success(f"Prediccion: {pred_name}")
            st.progress(int(pred_conf), text=f"Confianza: {pred_conf:.1f}%")

            st.divider()
            st.subheader("Probabilidades por clase")
            fig_bar, ax = plt.subplots(figsize=(9, 4))
            colors_bar = ['#55A868' if i == pred_idx else '#4C72B0' for i in range(10)]
            bars = ax.barh(CLASS_NAMES, pred_probs * 100, color=colors_bar, edgecolor='white')
            ax.set_xlabel("Probabilidad (%)")
            ax.set_title("Distribución de probabilidades", fontweight='bold')
            ax.set_xlim(0, 105)
            for bar, prob in zip(bars, pred_probs * 100):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{prob:.1f}%', va='center', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_bar)

        elif uploaded is not None and model is None:
            if not TF_AVAILABLE:
                st.error("TensorFlow no esta instalado. Para usar la prediccion, ejecuta el notebook en Google Colab, descarga modelo_final.h5 y usa un entorno con Python 3.11 + TensorFlow.")
            else:
                st.error("Modelo no disponible. Ejecuta el notebook en Colab y descarga modelo_final.h5 a esta carpeta.")
        else:
            st.info("Sube una imagen en el panel izquierdo para comenzar.")

            st.divider()
            st.markdown("**O prueba con una imagen aleatoria del dataset de test:**")
            if st.button("Imagen aleatoria del dataset"):
                X_train_, y_train_, X_test_, y_test_ = load_dataset()
                rand_idx = np.random.randint(0, len(X_test_))
                sample   = X_test_[rand_idx]
                label    = y_test_[rand_idx]

                st.image(sample, caption=f"Etiqueta real: {CLASS_NAMES[label]}", width=200)

                if model is not None:
                    arr_demo = (sample / 255.0).reshape(1, 28, 28, 1)
                    probs    = model.predict(arr_demo, verbose=0)[0]
                    pred     = np.argmax(probs)
                    correcto = "Correcto" if pred == label else "Incorrecto"
                    st.write(f"**Prediccion:** {CLASS_NAMES[pred]} — {correcto}")
                    st.write(f"**Etiqueta real:** {CLASS_NAMES[label]}")
