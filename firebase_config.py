import streamlit as st
import pyrebase

config = {
    "apiKey": st.secrets["firebase_web"]["API_KEY"],
    "authDomain": st.secrets["firebase_web"]["AuthDomain"],
    "projectId": st.secrets["firebase_web"]["ProjectId"],
    "storageBucket": st.secrets["firebase_web"]["StorageBucket"],
    "messagingSenderId": st.secrets["firebase_web"]["MessagingSenderId"],
    "appId": st.secrets["firebase_web"]["AppId"],
    "databaseURL": st.secrets["firebase_web"]["DatabaseURL"]
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()