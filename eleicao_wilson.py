# Avaliação A2 - Previsão de Eleição
# Com Base nos arquivos dos **dados eleitorais de 2020**, O Aluno deverá criar uma **Aplicação WEB** usando STREAMLIT para prever a situação do **Candidato a Vereador**, no município de **Campos de Goytacazes** (Eleito, Não eleito, Suplente)

# DETALHES
# 3 Datasets: https://www.tse.jus.br/eleicoes/estatisticas/repositorio-de-dados-eleitorais-1 > Prestação de contas eleitorais > 2020 > 3 Arquivos .zip
# Aluno: Wilson Pessanha Campinho

# pip install streamlit
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle
import sklearn.metrics as metrics
import streamlit as st
import tensorflow as tf
import zipfile
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

ZIP_PATH = 'A2Eleicoes.zip'
CSV_FILE1 = 'Wilson Pessanha Campinho - consulta_cand_2020_RJ.csv'
CSV_FILE2 = 'Wilson Pessanha Campinho - receitas_candidatos_2020_RJ.csv'
CSV_FILE3 = 'Wilson Pessanha Campinho - despesas_contratadas_candidatos_2020_RJ'
CHECKPOINT_DIR = './training_checkpoints'


def convert_data(dataframe):
    label_encoders = LabelEncoder()
    dataframe.iloc[:, 0] = label_encoders.fit_transform(dataframe.iloc[:, 0])

    for i in range(3, 9):
        dataframe.iloc[:, i] = label_encoders.fit_transform(
            dataframe.iloc[:, i])

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataframe[dataframe.columns] = scaler.fit_transform(
        dataframe[dataframe.columns])

    return dataframe


def load_data():
    """
    Carrega os dados do DataSet da eleição.
    :return: DataFrame com colunas selecionadas.
    """
    zip_object = zipfile.ZipFile(file=ZIP_PATH, mode='r')
    zip_object.extractall('./data')
    zip_object.close()

    candidatos = pd.read_csv(CSV_FILE1, sep=';', encoding="latin1")
    receitas = pd.read_csv(CSV_FILE2, sep=';', encoding="latin1")
    despesas = pd.read_csv(CSV_FILE3, sep=';', encoding="latin1")

    candidatos = candidatos.iloc[:, 0:23]
    candidatos.drop(['NM_URNA_CANDIDATO', 'SG_PARTIDO',
                    'VR_DESPESA_MAX_CAMPANHA'], axis=1, inplace=True)

    cargo = 'VEREADOR'
    municipio = 'CAMPOS DOS GOYTACAZES'

    df_candidato_mun = candidatos.loc[(candidatos['NM_UE'] == municipio) & (
        candidatos['DS_CARGO'] == cargo)]  # Candidatos de Campos
    df_receita_mun = receitas.loc[(receitas['NM_UE'] == municipio) & (
        receitas['DS_CARGO'] == 'Vereador')]  # Receitas de Candidatos de Campos
    df_despesa_mun = despesas.loc[(despesas['NM_UE'] == municipio) & (
        despesas['DS_CARGO'] == 'Vereador')]  # Despesas de Canditados de Campos

    df_receita_mun['VR_RECEITA'] = df_receita_mun['VR_RECEITA'].apply(
        lambda x: float(x.split()[0].replace(',', '.')))
    df_despesa_mun['VR_DESPESA_CONTRATADA'] = df_despesa_mun['VR_DESPESA_CONTRATADA'].apply(
        lambda x: float(x.split()[0].replace(',', '.')))

    soma_cand_receita = df_receita_mun[['NR_CPF_CANDIDATO', 'VR_RECEITA']].groupby(
        'NR_CPF_CANDIDATO').sum('VR_RECEITA')  # total receita por candidato
    soma_cand_despesa = df_despesa_mun[['NR_CPF_CANDIDATO', 'VR_DESPESA_CONTRATADA']].groupby(
        'NR_CPF_CANDIDATO').sum('VR_DESPESA_CONTRATADA')  # total despesa por Candidato
    despesa_receita = pd.merge(
        soma_cand_receita, soma_cand_despesa, on=['NR_CPF_CANDIDATO'])
    df_candidato_mun = pd.merge(
        candidatos, despesa_receita, on=['NR_CPF_CANDIDATO'])

    df_candidato_mun['DS_SIT_TOT_TURNO'].loc[df_candidato_mun['DS_SIT_TOT_TURNO']
                                             == 'ELEITO POR QP'] = 'ELEITO'
    df_candidato_mun['DS_SIT_TOT_TURNO'].loc[df_candidato_mun['DS_SIT_TOT_TURNO']
                                             == 'ELEITO POR MÉDIA'] = 'ELEITO'
    df_candidato_mun['DS_SIT_TOT_TURNO'].loc[df_candidato_mun['DS_SIT_TOT_TURNO']
                                             == '#NULO#'] = 'NÃO ELEITO'

    data = df_candidato_mun

    columns_to_drop = [
        'SG_UE', 'NM_UE', 'CD_CARGO', 'DS_CARGO', 'NR_CANDIDATO', 'NM_CANDIDATO', 'NR_CPF_CANDIDATO', 'NM_EMAIL', 'TP_AGREMIACAO', 'NR_PARTIDO', 'DS_COMPOSICAO_COLIGACAO'
    ]

    data.drop(columns=columns_to_drop, axis=1, inplace=True)
    dataset = data.copy()

    dataset = convert_data(dataset)

    return data, dataset


def load_x_and_y(data):
    """
    Usa o DataSet da eleição para devolver o X e Y de treino e teste.
    :return: X e Y do DataSet da eleição.
    """

    x = data.drop(columns='DS_SIT_TOT_TURNO', axis=1)
    y = data['DS_SIT_TOT_TURNO']

    return x, y


def build_model(x_shape, y_classes, rnn_units):
    print(f'Entrada: {x_shape}')

    model = Sequential()
    model.add(LSTM(rnn_units, input_shape=(x_shape[-1], 1)))
    model.add(Dense(y_classes))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


def load_ready_ai_model(x, y):
    y_classes = len(np.unique(y))
    rnn_units = 256

    checkpoint_prefix = os.path.join(CHECKPOINT_DIR)
    tf.train.latest_checkpoint(checkpoint_prefix)

    model = build_model(
        x_shape=x.shape, y_classes=y_classes, rnn_units=rnn_units)
    model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    model.build(tf.TensorShape([1, None]))

    return model


def load_sidebar(data):
    st.sidebar.header('Escolha os parâmetros para Predição')
    st.sidebar.subheader('Parâmetros')
    params = dict()

    dataset_name = st.sidebar.selectbox(
        'Selecione o Dataset', ['Consulta Candidatos 2020 RJ'])
    algorithm_name = st.sidebar.selectbox(
        'Selecione o Algoritmo', ['Rede Neural Recorrente'])

    nome_partidos = np.unique(data['NM_PARTIDO'])
    params['nome_partido'] = st.sidebar.selectbox(
        label='Nome do Partido', options=nome_partidos, key='nome_partido')

    cod_nacionalidade = np.unique(data['CD_NACIONALIDADE'])
    params['cod_nacionalidade'] = st.sidebar.selectbox(
        label='Código da nacionalidade (1 = Brasileira)', options=cod_nacionalidade, key='cod_nacionalidade')

    max_idade_posse = data['NR_IDADE_DATA_POSSE'].max()
    params['idade_posse'] = st.sidebar.number_input(
        'Idade na Posse', min_value=18, max_value=max_idade_posse, value=40, key='idade_posse')

    genero = np.unique(data['DS_GENERO'])
    params['genero'] = st.sidebar.selectbox(
        label='Gênero', options=genero, key='genero')

    escolaridade = np.unique(data['DS_GRAU_INSTRUCAO'])
    params['escolaridade'] = st.sidebar.selectbox(
        label='Escolaridade', options=escolaridade, key='escolaridade')

    estado_civil = np.unique(data['DS_ESTADO_CIVIL'])
    params['estado_civil'] = st.sidebar.selectbox(
        label='Estado Civil', options=estado_civil, key='estado_civil')

    tom_de_pele = np.unique(data['DS_COR_RACA'])
    params['tom_de_pele'] = st.sidebar.selectbox(
        label='Tom de pele', options=tom_de_pele, key='tom_de_pele')

    ocupacao = np.unique(data['DS_OCUPACAO'])
    params['ocupacao'] = st.sidebar.selectbox(
        label='Ocupação', options=ocupacao, key='ocupacao')

    max_valor_receita = data['VR_RECEITA'].max()
    params['valor_receita'] = st.sidebar.slider(
        label='Valor da Receita - R$', min_value=0.0, max_value=max_valor_receita, value=50000.0, key='valor_receita')

    max_valor_despesa = data['VR_DESPESA_CONTRATADA'].max()
    params['valor_despesa'] = st.sidebar.slider(
        label='Valor da Despesa Contratada - R$', min_value=0.0, max_value=max_valor_despesa, value=70000.0, key='valor_despesa')

    params = pd.DataFrame(params, index=[0])

    return dataset_name, algorithm_name, params


def main():
    election_data, election_dataset = load_data()
    dataset_name, algorithm_name, params = load_sidebar(data=election_data)

    st.title('Avaliação A2 - Previsão de Eleição')
    st.write('---')
    st.header(f"{dataset_name} Dataset")

    x_election, y_election = load_x_and_y(data=election_data)
    model = load_ready_ai_model(x=x_election.values, y=y_election.values)

    with open('eleicoes.pkl', 'rb') as f:
        x_train, x_test, y_train, y_test, scaler = pickle.load(f)

    prediction = model.predict(x_test)
    aux_table = pd.DataFrame()
    aux_table['DS_SIT_TOT_TURNO - Y Teste'] = y_test
    aux_table[f'DS_SIT_TOT_TURNO - {algorithm_name}'] = prediction
    prediction_percentage = metrics.r2_score(y_test, prediction)

    st.write('*Shape do Dataset*: ', election_data.shape)
    st.write('*Quantidade de Classes*: ', np.unique(y_election).size)
    st.write('*Algoritmo*: ', algorithm_name)
    st.write('*Precisão (%)*: ', prediction_percentage * 100)
    st.subheader('Começo da Tabela')
    st.write(election_data.head(3))
    st.write('---')

    st.header(f'Predição de {algorithm_name}')
    st.subheader('Status Candidatura')
    st.write(aux_table)

    st.subheader('Acurácia')
    evaluation = model.evaluate(x_test, y_test)
    st.write(f"*Perda*: {evaluation[0] * 100}%")
    st.write(f"*Acurácia*: {evaluation[1] * 100}%")

    st.header(f'Predição Manual (Ajustar valores ao lado)')
    st.write('*Shape Atual*: ', params.shape)
    st.write(params)
    st.subheader('Resultado obtido')
    converted_params = convert_data(params)
    manual_pred = model.predict(converted_params.values)
    man_pred_table = pd.DataFrame()
    man_pred_table['DS_SIT_TOT_TURNO - Customizado'] = manual_pred
    st.write(pd.DataFrame(man_pred_table))


if __name__ == '__main__':
    main()
