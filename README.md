# Reconhecimento de Emoções Faciais com Deep Learning

## Descrição

Este projeto foi desenvolvido para a disciplina de Deep Learning e tem como objetivo realizar o reconhecimento automático de emoções faciais utilizando Redes Neurais Profundas (Deep Learning).

Foram implementados e comparados quatro modelos de aprendizado profundo utilizando o dataset FER2013:

- DNN (Deep Neural Network)
- CNN (Convolutional Neural Network)
- MobileNetV2
- ResNet50

Além dos experimentos, foi desenvolvido um MVP utilizando Streamlit, permitindo ao usuário enviar uma imagem facial e receber a emoção prevista pelo modelo.

---

## Objetivos

- Implementar modelos de Deep Learning utilizando TensorFlow/Keras;
- Comparar o desempenho entre diferentes arquiteturas;
- Avaliar os modelos utilizando métricas de classificação;
- Desenvolver uma aplicação prática utilizando Streamlit;
- Armazenar as predições em banco de dados SQLite.

---

## Dataset

**FER2013 (Facial Expression Recognition 2013)**

O dataset contém aproximadamente 35 mil imagens faciais em escala de cinza, classificadas em sete emoções:

- 😠 Raiva (Angry)
- 🤢 Nojo (Disgust)
- 😨 Medo (Fear)
- 😊 Felicidade (Happy)
- 😐 Neutro (Neutral)
- 😢 Tristeza (Sad)
- 😲 Surpresa (Surprise)

---

## Tecnologias Utilizadas

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Streamlit
- SQLite

---

## Modelos Implementados

- Deep Neural Network (DNN)
- Convolutional Neural Network (CNN)
- MobileNetV2 (Transfer Learning)
- ResNet50 (Transfer Learning)

---

## Fluxo do Projeto

```text
Dataset FER2013
        │
        ▼
Leitura das Imagens
        │
        ▼
Pré-processamento
        │
        ▼
Normalização
        │
        ▼
Treinamento dos Modelos
        │
        ▼
Avaliação
        │
        ▼
MVP Streamlit
```

---

## Métricas Avaliadas

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de Confusão
- Curva ROC

---

## Resultados

| Modelo | Accuracy |
|---------|----------|
| DNN | 32,73% |
| CNN | 46,96% |
| ResNet50 | 41,67% |
| **MobileNetV2** | **55,64%** |

O MobileNetV2 apresentou o melhor desempenho entre os modelos avaliados, sendo utilizado no MVP desenvolvido em Streamlit.

---

## MVP

A aplicação permite:

- Upload de imagem facial;
- Classificação automática da emoção;
- Exibição da confiança da predição;
- Visualização das probabilidades de cada emoção;
- Registro das interações em banco SQLite.

---

## Estrutura do Projeto

```text
reconhecimento-emocoes-faciais-deep-learning
│
├── artigo/
├── apresentacao/
├── notebook-colab/
├── mvp-emocoes/
│   ├── app.py
│   ├── criar_banco.py
│   ├── requirements.txt
│   └── modelo_mobilenetv2_emocoes.keras
├── imagens/
├── resultados/
├── README.md
└── .gitignore
```

---

## Como Executar

Clone o repositório:

```bash
git clone https://github.com/RuthLima15/reconhecimento-emocoes-faciais-deep-learning.git
```

Entre na pasta do projeto:

```bash
cd reconhecimento-emocoes-faciais-deep-learning
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação:

```bash
streamlit run app.py
```

---

## Artigos Utilizados

- AKHAND, M. A. H. et al. *Facial Emotion Recognition Using Transfer Learning in the Deep CNN*. Electronics, 2021.

- COSTA, N. E. A. et al. *Reconhecimento de expressões faciais utilizando redes neurais convolucionais*. COBICET, 2024.

- LIMA, P. V. B. O. *Detecção Automática de Emoções no Aprendizado de Algoritmos*. UFMA, 2023.

---

## Autora

**Ruth Maria de Lima**

Curso: Análise e Desenvolvimento de Sistemas

Instituto Federal de Pernambuco – IFPE
