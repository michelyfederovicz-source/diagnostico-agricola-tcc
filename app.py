import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime
import os

MODEL_PATH = "modelo/modelo_mobilenet.h5"
DB_PATH = "banco/diagnostico_agricola.db"

st.set_page_config(page_title="Diagnóstico Agrícola IA", page_icon="🌱")

st.title("🌱 Diagnóstico Agrícola com IA")
st.markdown("TCC - Michely Federovicz")

if not os.path.exists(MODEL_PATH):
    st.error("Modelo não encontrado. Treine o modelo primeiro.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

classes = [
"Planta saudável",
"Pulgões",
"Lagartas",
"Ácaros",
"Estresse hídrico",
"Formiga cortadeira"
]

recomendacoes = {
"Planta saudável": """
Situação identificada:
A planta apresenta bom desenvolvimento, coloração adequada e ausência de sinais visíveis de pragas, doenças ou estresse hídrico.

O que fazer agora:
- Manter o manejo atual da cultura.
- Realizar inspeções visuais semanais.
- Remover folhas secas ou danificadas.
- Manter irrigação e adubação equilibradas.

Boas práticas preventivas:
- Rotação de culturas.
- Uso de cobertura vegetal no solo.
- Manutenção da limpeza da área.

Observação:
O monitoramento contínuo ajuda a prevenir problemas futuros.
""",

"Pulgões": """
O que isso causa: Os pulgões sugam a seiva da planta, enfraquecendo seu crescimento e podendo causar deformação das folhas.

O que fazer agora (ações imediatas):
    - Lavar as folhas com água para remover os insetos.
    - Remover manualmente folhas muito infestadas.
    - Evitar excesso de adubação nitrogenada.

Manejo de baixo risco:
    - Utilizar soluções caseiras de sabão neutro diluído em água.
    - Aplicar óleo vegetal diluído para dificultar a fixação dos insetos.
    - Incentivar inimigos naturais, como joaninhas.

Atenção: Caso a infestação aumente rapidamente, recomenda-se buscar orientação técnica local (extensão rural ou cooperativa).
""",

"Lagartas":"""
O que isso causa: As lagartas se alimentam das folhas, reduzindo a área fotossintética e prejudicando o desenvolvimento da planta.

O que fazer agora:
    - Inspecionar as plantas no início da manhã ou final da tarde.
    - Retirar lagartas manualmente sempre que possível.
    - Remover folhas muito danificadas.

Boas práticas de manejo:
    - Manter a área limpa, sem restos de cultura.
    - Utilizar barreiras físicas simples.
    - Realizar monitoramento frequente da lavoura.

Atenção: Em casos de infestação intensa, buscar apoio técnico especializado.
""",

"Ácaros":"""
O que isso causa: Os ácaros provocam manchas amareladas, aspecto ressecado e reduzem a capacidade fotossintética da planta.

O que fazer agora:
    - Aumentar a umidade do ambiente quando possível.
    - Lavar as folhas para reduzir a população inicial.
    - Remover folhas severamente atacadas.

Manejo preventivo:"
    - Evitar estresse hídrico da planta.
    - Reduzir poeira no ambiente.
    - Manter irrigação adequada.

Atenção: Se os sintomas persistirem, recomenda-se buscar orientação técnica.
""",

"Estresse hídrico":"""
O que isso causa: Falta ou excesso de água pode causar murcha, amarelecimento e redução do crescimento da planta.

O que fazer agora:
    - Verificar a umidade do solo antes de irrigar.
    - Ajustar a frequência e o volume de irrigação.
    - Evitar irrigar nos horários mais quentes do dia.

Boas práticas:
    - Utilizar cobertura vegetal (palhada).
    - Melhorar a retenção de água no solo.
    - Evitar compactação do solo.

Observação: Pequenos ajustes no manejo da água podem recuperar a planta rapidamente.
""",

"Formiga cortadeira":"""
Situação identificada: As formigas cortadeiras (gêneros Atta spp. - saúvas, e Acromyrmex spp. - quenquéns) cortam folhas, brotos, flores e partes tenras das plantas para cultivar um fungo dentro do ninho. Isso causa desfolha rápida, enfraquecimento da planta, redução da fotossíntese e, em mudas jovens ou plantas pequenas, pode levar à morte.

O que isso causa: Perda de área foliar, redução do crescimento, maior suscetibilidade a outras pragas e doenças, prejuízo em lavouras novas, pomares, hortas e reflorestamentos.

O que fazer agora (ações imediatas):
    - Inspecionar a área ao redor das plantas atacadas e localizar os formigueiros ativos (procure trilhas de formigas e montes de terra solta).
    - Proteger plantas jovens com barreiras físicas: cones invertidos de plástico ou lata no tronco (pinte ou passe graxa/vaselina na borda interna para impedir subida).
    - Remover manualmente folhas cortadas e restos próximos ao ninho para reduzir o material disponível.
    - Evitar deixar restos vegetais acumulados perto das plantas.

Manejo de baixo risco e preventivo:
    - Incentivar inimigos naturais: aves (como pica-paus, joão-de-barro), tamanduás, tatus e lagartos que comem formigas.
    - Plantar espécies repelentes ao redor da lavoura: batata-doce, gergelim, rim-de-boi, capim-limão, hortelã, citronela ou eucalipto (em bordadura).
    - Melhorar o solo com matéria orgânica (composto, cobertura morta) para aumentar a biodiversidade e reduzir atratividade.
    - Manter a área limpa de restos de poda e capina, evitando acúmulo de material vegetal fresco.
    - Em pequenas áreas: usar armadilhas com iscas naturais (ex: folhas de laranja ou extratos de plantas como nim misturado com óleo) para desviar o forrageamento.

Atenção: Em infestações grandes ou em lavouras comerciais, o monitoramento constante é essencial. Caso o ataque persista ou cause perdas significativas, recomenda-se buscar orientação técnica local (Embrapa, Emater, extensionistas rurais ou cooperativa agrícola) para avaliar manejo integrado. Evite uso indiscriminado de formicidas químicos para preservar inimigos naturais e saúde do solo.

Observação: O manejo integrado e preventivo é a melhor estratégia a longo prazo. Formigas cortadeiras fazem parte do ecossistema e só viram praga em áreas desequilibradas (monoculturas, solo pobre). Pequenas ações culturais podem reduzir drasticamente os danos sem agredir o ambiente.
"""
}

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS diagnosticos (
id INTEGER PRIMARY KEY AUTOINCREMENT,
nome_imagem TEXT,
praga TEXT,
confianca REAL,
recomendacao TEXT,
data_analise TEXT
)
""")

conn.commit()

st.subheader("Envie uma imagem da planta")

uploaded_file = st.file_uploader(
"Selecione a imagem",
type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img)

    if st.button("Analisar"):

        with st.spinner("Analisando imagem..."):

            img = img.resize((224,224))
            img_array = np.array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)[0]

            idx = np.argmax(pred)

            praga = classes[idx]
            conf = pred[idx]*100

            recomendacao = recomendacoes.get(praga,"Sem recomendação.")

            data = datetime.now().strftime("%d/%m/%Y %H:%M")

            cursor.execute(
                "INSERT INTO diagnosticos VALUES (NULL,?,?,?,?,?)",
                (uploaded_file.name,praga,float(conf),recomendacao,data)
            )

            conn.commit()

            st.success(f"Diagnóstico: {praga}")
            st.write(f"Confiança: {conf:.2f}%")
            st.write("Recomendação:")
            st.write(recomendacao)

st.markdown("---")
st.subheader("Histórico de diagnósticos")

cursor.execute("SELECT * FROM diagnosticos ORDER BY id DESC LIMIT 5")
rows = cursor.fetchall()

for row in rows:
    st.write(f"{row[5]} - {row[2]} ({float(row[3]):.2f}%)")