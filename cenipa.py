import streamlit as st
import pandas as pd
import pickle as pk
from sklearn.preprocessing import LabelEncoder
import math

# Retorna o dataframe original
def read_data(path):

    data = pd.read_csv(path)

    return data

# Retorna o modelo ja treinado
def read_model(path):

    with open(path, mode = 'rb') as f:
        model = pk.load(f)

    return model

# Roda o LabelEncoder no dataframe original para equivaler o modelo treinado
def encode(data):

    label_enc = LabelEncoder()

    for i in (1,2,4,7,11,15,17,21,22):

        data.iloc[:,i] = label_enc.fit_transform(data.iloc[:,i])

    return data

# Remove o Y e algumas colunas com valores todos iguais para equivaler o modelo treinado
def trim_x(data):
    
    X_data = data.iloc[:,:]
    X_data = X_data.drop('Attrition', axis = 1)
    X_data = X_data.drop('EmployeeCount', axis = 1)
    X_data = X_data.drop('Over18', axis = 1)
    X_data = X_data.drop('StandardHours', axis = 1)

    return X_data

# Cria os sliders para o usuario pode escolher os dados a serem usados na previsão
def user_input(X_data):
    
    data = {}

    for col in X_data.columns.to_list():
    
      X_data[col] = X_data[col].astype(float, errors = 'raise')
      data[col] = st.sidebar.slider(col, X_data[col].min(), X_data[col].max(), float(math.floor(X_data[col].mean())), 1.0)
          
    return pd.DataFrame(data, index=[0])

# Definindo o main
def main():

    # Lendo o dataframe original e o modelo treinado
    data = read_data('./Human_Resources.csv')
    model = read_model('emp_att.pkl')

    # Rodando o encode e podando o X
    data = encode(data)
    X_data = trim_x(data)

    # Estrutura do site
    st.write(""" # Predição de um empregado possuir atrito # """)
    
    st.write('---')
   
    st.sidebar.header('Escolha de paramentros para Predição')

    df = user_input(X_data)

    st.header('Parametros especificados')

    st.write(df)

    st.write('---')

    prediction = model.predict(df)

    st.header('Attrition Prevista')

    st.write(prediction)

    st.write('---')

# Rodando o main
if __name__=='__main__':
    
    main()
