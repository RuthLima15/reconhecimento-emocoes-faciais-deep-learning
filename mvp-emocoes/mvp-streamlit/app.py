import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


st.set_page_config(
    page_title="Reconhecimento de Emoções Faciais",
    page_icon="🙂",
    layout="wide"
)

CLASS_NAMES = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

TRADUCAO = {
    "angry": "Raiva",
    "disgust": "Nojo",
    "fear": "Medo",
    "happy": "Felicidade",
    "neutral": "Neutro",
    "sad": "Tristeza",
    "surprise": "Surpresa"
}

EMOJIS = {
    "Raiva": "😠",
    "Nojo": "🤢",
    "Medo": "😨",
    "Felicidade": "😊",
    "Neutro": "😐",
    "Tristeza": "😢",
    "Surpresa": "😲"
}


@st.cache_resource
def carregar_modelo():
    caminho_modelo = os.path.join(
        os.path.dirname(__file__),
        "modelo_mobilenetv2_emocoes.keras"
    )
    return tf.keras.models.load_model(caminho_modelo)


def criar_banco():
    conexao = sqlite3.connect("interacoes.db")
    cursor = conexao.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predicoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            emocao_prevista TEXT,
            confianca REAL
        )
    """)

    conexao.commit()
    conexao.close()


def salvar_interacao(emocao, confianca):
    conexao = sqlite3.connect("interacoes.db")
    cursor = conexao.cursor()

    cursor.execute("""
        INSERT INTO predicoes (data_hora, emocao_prevista, confianca)
        VALUES (?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        emocao,
        confianca
    ))

    conexao.commit()
    conexao.close()


def listar_interacoes():
    conexao = sqlite3.connect("interacoes.db")
    df = pd.read_sql_query(
        "SELECT * FROM predicoes ORDER BY id DESC",
        conexao
    )
    conexao.close()
    return df


def preparar_imagem(imagem):
    imagem = imagem.convert("RGB")
    imagem = imagem.resize((224, 224))  # use 96x96 se seu modelo for 96

    img_array = np.array(imagem)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


criar_banco()
modelo = carregar_modelo()


st.title("Reconhecimento de Emoções Faciais com Deep Learning")

st.markdown("""
Este MVP utiliza o modelo **MobileNetV2** para identificar emoções faciais a partir de uma imagem enviada pelo usuário.

O sistema classifica a imagem em uma das seguintes emoções: **raiva, nojo, medo, felicidade, neutro, tristeza ou surpresa**.
""")


col_upload, col_info = st.columns([1, 1])

with col_upload:
    arquivo = st.file_uploader(
        "Envie uma imagem de rosto",
        type=["jpg", "jpeg", "png"]
    )

with col_info:
    st.info("""
    **Como usar:**

    1. Envie uma imagem facial.
    2. O modelo fará a predição.
    3. O resultado será exibido na tela.
    4. A interação será salva no banco SQLite.
    """)


if arquivo is not None:
    imagem = Image.open(arquivo).convert("RGB")

    st.divider()
    st.subheader("Imagem enviada e resultado")

    col_img, col_resultado = st.columns([1, 1])

    with col_img:
        st.image(
            imagem,
            caption="Imagem enviada pelo usuário",
            width=350
        )

    with col_resultado:
        img_processada = preparar_imagem(imagem)
        predicao = modelo.predict(img_processada)

        indice = np.argmax(predicao)
        confianca = float(np.max(predicao))

        emocao = CLASS_NAMES[indice]
        emocao_pt = TRADUCAO[emocao]

        salvar_interacao(emocao_pt, confianca)

        st.success("Predição realizada com sucesso!")

        st.metric(
            "Emoção Detectada",
            f"{EMOJIS[emocao_pt]} {emocao_pt}"
        )

        if confianca >= 0.80:
            st.success(f"Confiança alta: {confianca:.2%}")
        elif confianca >= 0.60:
            st.info(f"Confiança moderada: {confianca:.2%}")
        else:
            st.warning(
                f"Confiança baixa: {confianca:.2%}. "
                "O resultado pode não representar corretamente a emoção detectada."
            )

        st.subheader("Top 3 emoções detectadas")

        top3 = np.argsort(predicao[0])[::-1][:3]

        for i in top3:
            emocao_top = TRADUCAO[CLASS_NAMES[i]]
            st.write(
                f"{EMOJIS[emocao_top]} "
                f"**{emocao_top}** - "
                f"{predicao[0][i]:.2%}"
            )

    st.subheader("Probabilidade de cada emoção")

    df_prob = pd.DataFrame({
        "Emoção": [TRADUCAO[c] for c in CLASS_NAMES],
        "Probabilidade": predicao[0]
    }).sort_values("Probabilidade", ascending=False)

    st.dataframe(df_prob, use_container_width=True)

    df_prob_grafico = df_prob.set_index("Emoção")
    st.bar_chart(df_prob_grafico)


st.divider()
st.subheader("Histórico das Predições")

df = listar_interacoes()

if len(df) > 0:
    st.dataframe(df, use_container_width=True)
else:
    st.info("Nenhuma predição realizada até o momento.")