"""Streamlit app to generate Tweets."""

# Import from standard library
import logging
from math import ceil, floor
import random
import re

# Import from 3rd party libraries
import streamlit as st
import streamlit.components.v1 as components
import streamlit_analytics
import pickledb

# Configure logger
logging.basicConfig(format="\n%(asctime)s\n%(message)s", level=logging.INFO, force=True)

# Configure Streamlit page and state
st.set_page_config(page_title="CompteGPU", page_icon="💻")

# Force responsive layout for columns also on mobile
st.write(
    """<style>
    [data-testid="column"] {
        width: calc(50% - 1rem);
        flex: 1 1 calc(50% - 1rem);
        min-width: calc(50% - 1rem);
    }
    </style>""",
    unsafe_allow_html=True,
)

# Render Streamlit page
#streamlit_analytics.start_tracking()
st.title("Calculer le nombre de GPU nécessaire pour inférence LLM")
st.markdown(
"""
Ce projet vous aide à calculer le nombre de GPU nécessaires pour faire tourner des modèles de langage (LLM) en inférence.

Les LLM comme GPT-4 nécessitent beaucoup de ressources de calcul. Cette application vous permet d'estimer rapidement vos besoins en GPU en fonction de différents paramètres :

- Taille du modèle (paramètres) en milliards
- Précision du modèle (8bits, 16 bits ou 32 bits)
- Mémoire VRAM par GPU en Go
- FLOPS par GPU (en Teraflops)
- FLOPs par token (en TeraFLOPs)
- Nombre d'utilisateurs simultanés
- Débit par utilisateur (en tokens par seconde)
- Batch maximal (ex : 10 requêtes par batch)
//- Latence maximale acceptable (en ms)


//- La taille du modèle que vous souhaitez utiliser
//- Le nombre de requêtes par seconde à traiter
//- La latence maximale acceptable
//- Votre budget

Remplissez simplement le formulaire ci-dessous avec vos besoins et nous vous fournirons une estimation du nombre de GPU requis ainsi que les coûts associés.

"""
)

#renting cost Outscale A100 80gb = 2k€/mois donc 2.74€/h HT
#chez scaleway H100 80gb = 2.73€/h
#outscale prix A100 80gb = 3.6€/h (HT probablement)
#on a acheté 16 H100 80gb pour à peu près 30k€ l'unité (31875€).
#censé être rentabilisé en 16 mois (1 an et 4 mois)
memoire_gpu_dict = {"A100_40gb": 40, "H100_80gb": 80}
flops_gpu_dict = {
    "A100_40gb": 
    {
        "8 bits (int8/fp8)": 624,
        "16 bits (fp16)": 312,
        "32 bits (fp32)": 19.5 #(mais TF32 156)
    }, 
    "H100_80gb": #attention différence entre SXM et NVL (SXM plus rapide)
    {
        "8 bits (int8/fp8)": 3341, #NVL et SXM=3958 #fp8
        "16 bits (fp16)": 1671, #NVL et SXM=1979
        "32 bits (fp32)": 60, #SXM = 67 mais TF32 835 et SXM = 989 
    }
}
#attention car la vitesse GPU dépend en fait aussi de la précision du modèle TODO

methode = st.selectbox(label="Méthode de calcul", options=["Batching", "Multi-modèle"])
if methode == "Batching":
    which_model = st.selectbox(label="Quel modèle ?", options=["Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct"])
    taille_modele = st.number_input(label="Taille du modèle (en milliards)", value=70)
    precision_modele = st.selectbox(label="Précision du modèle", options=["8 bits (int8/fp8)", "16 bits (fp16)", "32 bits (fp32)"]) #en fait les modeles sont en fp16 par defaut
    quel_gpu = st.selectbox(label="Quel GPU ?", options=["A100_40gb", "H100_80gb"]) #, "A1000", "A800", "A4000"])
    #memoire_gpu = st.number_input(label="Mémoire VRAM par GPU (en Go)", value=80)
    #flops_gpu = st.number_input(label="FLOPS par GPU (en Teraflops)", value=19.5)
    #flops_token = st.number_input(label="FLOPs par token (en TeraFLOPs)", value=70*2*10**9/10**12)
    utilisateurs_simultanes = st.number_input(label="Nombre d'utilisateurs simultanés", value=100)
    debit_utilisateur = st.number_input(label="Débit par utilisateur (en tokens par seconde)", value=10)
    batch_maximal = st.number_input(label="Batch maximal (ex : 10 requêtes par batch)", value=10)
    MAX_TOKEN_BATCH = st.number_input(label="Nombre de tokens par requête dans le batch", value=1000)
    #latence_maximale = st.number_input(label="Latence maximale acceptable (en ms)", value=100)
else:
    taille_modele = st.number_input(label="Taille du modèle (en milliards)", value=70)
    precision_modele = st.selectbox(label="Précision du modèle", options=["8 bits (int8/fp8)", "16 bits (fp16)", "32 bits (fp32)"])
    memoire_gpu = st.number_input(label="Mémoire VRAM par GPU (en Go)", value=80)
    flops_gpu = st.number_input(label="FLOPS par GPU (en Teraflops)", value=19.5)
    flops_token = st.number_input(label="FLOPs par token (en TeraFLOPs)", value=70*2*10**9/10**12)
    utilisateurs_simultanes = st.number_input(label="Nombre d'utilisateurs simultanés", value=100)
    debit_utilisateur = st.number_input(label="Débit par utilisateur (en tokens par seconde)", value=10)
    #batch_maximal = st.number_input(label="Batch maximal (ex : 10 requêtes par batch)", value=10)
    #latence_maximale = st.number_input(label="Latence maximale acceptable (en ms)", value=100)

precision = {"8 bits (int8/fp8)": 1, "16 bits (fp16)": 2, "32 bits (fp32)": 4}
#MAX_TOKEN_BATCH = 1000

dim_embedding_dic = {"Llama-3.1-8B-Instruct": 4096, "Llama-3.1-70B-Instruct": 128000}

# lancer le calcul
st.session_state.calcul_lance = st.button(label="Calculer", type="primary")
if st.session_state.calcul_lance:
    # TODO : faire le calcul
    if methode == "Batching":  
        memoire_gpu = memoire_gpu_dict[quel_gpu]
        flops_gpu = flops_gpu_dict[quel_gpu][precision_modele]
        flops_token = taille_modele * 2 * (10**9) / (10**12) #en TeraFLOPs
        #todo : intégrer kv caching 
        memoire_modele = taille_modele * precision[precision_modele] #en Go
        dim_embedding = dim_embedding_dic[which_model] #4096 #TODO
        memoire_tokens = batch_maximal * MAX_TOKEN_BATCH * dim_embedding * precision[precision_modele] / (2**30) #en Go
        memoire_kv = 320 * MAX_TOKEN_BATCH * batch_maximal / (2**20) #TODO la formule actuelle est pour llama2
        memoire_allouee_batch = memoire_modele + memoire_tokens + memoire_kv
        #attention à tout bien compter (activations etc)
        memory_constraint = memoire_allouee_batch / memoire_gpu
        #temps_pour_un_token = flops_token / flops_gpu
        combien_de_tours = ceil(utilisateurs_simultanes / batch_maximal)
        quantite_de_calcul_pour_un_tour_un_token = flops_token * batch_maximal
        temps_pour_un_tour_un_token = quantite_de_calcul_pour_un_tour_un_token / flops_gpu
        debit_config = 1/temps_pour_un_tour_un_token
        if(debit_utilisateur > debit_config):
            print("Batch size trop grand")
            nombre_gpu_necessaire = -1
        else:
            #idée naive : utiliser combien_de_tours GPU
            #amélioration : faire tourner un gpu sur plusieurs tours tant qu'on tient dans le débit cible
            speed_constraint = combien_de_tours / floor(debit_config / debit_utilisateur)
            #speed_constraint = utilisateurs_simultanes * debit_utilisateur / (batch_maximal * (flops_gpu / flops_token))
            nombre_gpu_necessaire_batching = max(memory_constraint, speed_constraint)
            nombre_gpu_necessaire = nombre_gpu_necessaire_batching
    else:
        nombre_gpu_necessaire_multimodel = utilisateurs_simultanes * debit_utilisateur / ((memoire_gpu / (taille_modele * precision[precision_modele])) *  flops_gpu / flops_token)
        nombre_gpu_necessaire = nombre_gpu_necessaire_multimodel #max(nombre_gpu_necessaire_batching, nombre_gpu_necessaire_multimodel)
    
    # affiche le résultat
    st.write(f"Nombre de GPU nécessaire : {nombre_gpu_necessaire}")


#streamlit_analytics.stop_tracking()