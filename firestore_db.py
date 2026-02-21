import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_service_account"]))
    firebase_admin.initialize_app(cred)
else:
    firebase_admin.get_app()

db = firestore.client()