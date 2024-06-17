import streamlit as st

def menu():
    #Muestra el nuevo menú
    st.sidebar.page_link("aplicacion.py", label="Inicio")

menu()
st.title("Entrenamiento con MNIST")

row = st.columns(2)
t1 = row[0].container(height=70, border= False)
t1.link_button("Algoritmos clásicos", "http://localhost:8501/AlgoritmosC")
t2 = row[1].container(height=70, border= False)
t2.link_button("Redes Neuronales", "http://localhost:8501/RedesN")