# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk import download as nltk_download
import re
import unicodedata
import spacy
from transformers import pipeline
from nltk.util import ngrams
from io import StringIO

# ---------------------------
# Configuración de página
# ---------------------------
st.set_page_config(
    page_title="Análisis de Opiniones (Videojuegos)",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Análisis de Opiniones de Videojuegos (ES)")

# ---------------------------
# Utilidades / Cache
# ---------------------------
@st.cache_resource(show_spinner="Cargando modelo de spaCy (es_core_news_sm)…")
def cargar_spacy():
  
    return spacy.load("es_core_news_sm")

@st.cache_resource(show_spinner="Cargando modelo de sentimiento (Hugging Face)…")
def cargar_modelo_sentimiento():
 
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

@st.cache_resource(show_spinner="Cargando stopwords de NLTK…")
def cargar_stopwords_es():
    try:
        _ = stopwords.words('spanish')
    except LookupError:
        nltk_download('stopwords')
    return set(stopwords.words('spanish'))

nlp = cargar_spacy()
modelo_sentimiento = cargar_modelo_sentimiento()
STOP_ES = cargar_stopwords_es()

# ---------------------------
# Funciones NLP
# ---------------------------
def normalizar_minusculas_sin_tildes(texto: str) -> str:
    t = (texto or "").lower()
    # quitar tildes
    t = ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')
    return t

def lematizar_filtrar(texto: str):
  
    texto = normalizar_minusculas_sin_tildes(str(texto))
    doc = nlp(texto)
    tokens = [
        t.lemma_ for t in doc
        if t.is_alpha and (t.lemma_ not in STOP_ES) and len(t.lemma_) > 2
    ]
    return tokens

def clasificar_etiqueta(label_hf: str) -> str:
    # Mapea etiquetas del modelo a Positivo / Neutral / Negativo
    if label_hf in ("Very Positive", "Positive"):
        return "Positivo"
    if label_hf in ("Very Negative", "Negative"):
        return "Negativo"
    return "Neutral"

def clasificar_sentimiento(texto: str) -> str:
    try:
        out = modelo_sentimiento(texto)
        if out and isinstance(out, list) and len(out) > 0:
            return clasificar_etiqueta(out[0]["label"])
    except Exception:
        pass
    return "Neutral"

def respuesta_por_sentimiento(s: str) -> str:
    if s == "Positivo":
        return "¡Gracias por tu reseña positiva! 😊 ¿Qué fue lo que más te gustó?"
    if s == "Negativo":
        return "Lamentamos tu experiencia 😔. ¿Podrías darnos más detalles para ayudarte?"
    return "Gracias por tu comentario. ¿Podrías ampliar un poco más tu experiencia?"

# ---------------------------

# ---------------------------
with st.sidebar:
    st.header("🧭 Pasos")
    st.markdown(
        "1. **Subir** un **CSV** con la columna **`opinion`** (mínimo 20 filas)\n"
        "2. **Ver** la nube de palabras y el **Top-10 de palabras principales**\n"
        "3. **Clasificar** sentimientos y **ver** el porcentaje por clase\n"
        "4. **Escribir** un comentario nuevo para **analizar** su tono"
    )

# ---------------------------
# Carga de datos
# ---------------------------
archivo = st.file_uploader("📤 Sube tu archivo CSV (columna obligatoria: `opinion`)", type=["csv"])

df = None
if archivo is not None:
    
    content_preview = archivo.read().decode("utf-8", errors="ignore")
    st.text_area("Vista previa del archivo (primeros 1000 caracteres)", content_preview[:1000], height=150)
    #
    archivo.seek(0)
    read_ok = False
    for sep in [",", ";", "\t", "|"]:
        try:
            df_try = pd.read_csv(archivo, sep=sep)
            if "opinion" in df_try.columns:
                df = df_try
                read_ok = True
                break
        except Exception:
            archivo.seek(0)
            continue
    if not read_ok:
       
        archivo.seek(0)
        try:
            df = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"Error al leer el CSV: {e}")
            df = None

# ---------------------------
# Validaciones de datos
# ---------------------------
if df is not None:
    if "opinion" not in df.columns:
        st.error("El CSV debe incluir una columna llamada **`opinion`**.")
        st.stop()
    if len(df) < 20:
        st.warning("El taller exige **al menos 20 opiniones**. Sube un CSV con 20 o más filas.")
        st.stop()

    # Mostrar muestra de datos
    st.subheader("👀 Vista de datos")
    st.dataframe(df[["opinion"]].head(10), use_container_width=True)

    # ---------------------------
    # Procesamiento del texto 
    # ---------------------------
    st.subheader("🔧 Procesamiento del texto")
    with st.spinner("Procesando texto…"):
        tokens_total = []
        for op in df["opinion"].astype(str).tolist():
            tokens_total.extend(lematizar_filtrar(op))

    if len(tokens_total) == 0:
        st.warning("Tras el procesamiento, no quedaron palabras válidas. Revisa que las opiniones estén en español.")
        st.stop()

    # ---------------------------
    # Nube de palabras
    # ---------------------------
    st.subheader("☁️ Nube de palabras")
    try:
        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
        nube = WordCloud(width=1000, height=400, background_color="white").generate(" ".join(tokens_total))
        ax_wc.imshow(nube, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc, use_container_width=True)
    except Exception as e:
        st.info(f"No se pudo generar la nube de palabras: {e}")

    # ---------------------------
    # Top-10 palabras principales 
    # ---------------------------
    st.subheader("📊 Top-10 palabras principales")
    top_lemmas = Counter(tokens_total).most_common(10)
    if top_lemmas:
        etiquetas, cuentas = zip(*top_lemmas)
        fig_top, ax_top = plt.subplots(figsize=(8, 4))
        ax_top.barh(etiquetas, cuentas)
        ax_top.set_xlabel("Frecuencia")
        ax_top.set_title("Top-10 palabras")
        ax_top.invert_yaxis()
        st.pyplot(fig_top, use_container_width=True)
    else:
        st.info("No hay palabras suficientes para el Top-10.")

    # ---------------------------
    # Clasificación de sentimiento
    # ---------------------------
    st.subheader("🧠 Clasificación de sentimiento (Positivo / Neutral / Negativo)")
    with st.spinner("Clasificando opiniones…"):
        df["Sentimiento"] = df["opinion"].astype(str).apply(clasificar_sentimiento)

    st.dataframe(df[["opinion", "Sentimiento"]], use_container_width=True)

    # Conteo y porcentaje por clase
    conteo = df["Sentimiento"].value_counts()
    pct = (conteo / conteo.sum() * 100).round(1)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Conteo por clase**")
        fig_c, ax_c = plt.subplots(figsize=(6, 3))
        ax_c.barh(conteo.index, conteo.values)
        ax_c.set_xlabel("Cantidad")
        ax_c.set_title("Distribución de opiniones")
        ax_c.invert_yaxis()
        st.pyplot(fig_c, use_container_width=True)
    with col2:
        st.markdown("**Porcentaje por clase**")
        fig_p, ax_p = plt.subplots(figsize=(6, 3))
        ax_p.barh(pct.index, pct.values)
        ax_p.set_xlabel("%")
        ax_p.set_title("Porcentaje de opiniones por clase")
        ax_p.invert_yaxis()
        st.pyplot(fig_p, use_container_width=True)

    # ---------------------------
    # NUEVO: Palabras clave por sentimiento (reemplaza a bigramas)
    # ---------------------------
    st.subheader("💬 Palabras clave por sentimiento")
    # tokenizar por fila y agregar por clase
    tokens_por_clase = {"Positivo": [], "Neutral": [], "Negativo": []}
    for _, row in df[["opinion", "Sentimiento"]].iterrows():
        clase = row["Sentimiento"]
        toks = lematizar_filtrar(row["opinion"])
        if clase in tokens_por_clase:
            tokens_por_clase[clase].extend(toks)

    cols = st.columns(3)
    orden = ["Positivo", "Neutral", "Negativo"]
    for i, clase in enumerate(orden):
        with cols[i]:
            st.markdown(f"**{clase}**")
            pares = Counter(tokens_por_clase[clase]).most_common(8)
            if pares:
                etiquetas_c, cuentas_c = zip(*pares)
                fig_k, ax_k = plt.subplots(figsize=(6, 3))
                ax_k.barh(etiquetas_c, cuentas_c)
                ax_k.set_xlabel("Frecuencia")
                ax_k.set_title(f"{clase}")
                ax_k.invert_yaxis()
                st.pyplot(fig_k, use_container_width=True)
            else:
                st.info("Sin suficientes palabras para mostrar.")

    # ---------------------------
    # Filtro por clase (opcional)
    # ---------------------------
    st.subheader("🔎 Filtrar opiniones por clase")
    clase_sel = st.multiselect("Selecciona clase(s) para ver:", options=["Positivo", "Neutral", "Negativo"], default=[])
    if clase_sel:
        st.dataframe(df[df["Sentimiento"].isin(clase_sel)][["opinion", "Sentimiento"]], use_container_width=True)

    # ---------------------------
    # Descargar resultados
    # ---------------------------
    st.subheader("💾 Descargar resultados")
    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False, encoding="utf-8")
    st.download_button(
        label="Descargar CSV con sentimientos",
        data=csv_buf.getvalue(),
        file_name="opiniones_con_sentimiento.csv",
        mime="text/csv"
    )

    # ---------------------------
    # Analizar un comentario nuevo
    # ---------------------------
    st.subheader("🧪 Analizar un comentario nuevo")
    nuevo = st.text_area("Escribe una opinión nueva (español):", height=90, placeholder="Ej.: El rendimiento es muy malo en PC, se calienta demasiado…")
    if st.button("Analizar comentario"):
        if nuevo.strip():
            senti = clasificar_sentimiento(nuevo)
            st.write(f"**Sentimiento estimado:** {senti}")
            st.info(respuesta_por_sentimiento(senti))
        else:
            st.warning("Por favor escribe una opinión para analizar.")
else:
    st.info("Carga un archivo .csv con una columna llamada **`opinion`** para comenzar.")
