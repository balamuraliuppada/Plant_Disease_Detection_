import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from utils import register_user, authenticate_user, save_scan, get_user_history
import base64

def get_base64_img(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="LeafSense AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================= THEME CONSTANTS =================
PRIMARY_COLOR = "#10b981"
BG_COLOR = "#f8fafc"
TEXT_COLOR = "#1e293b"

# ================= CSS STYLING =================
def load_css():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background-color: {BG_COLOR};
        }}
        
        /* Reduce default streamlit top padding */
        .block-container {{
            padding-top: 3rem;
        }}

        /* --- DASHBOARD GALLERY CARDS --- */
        .selection-card {{
            background: white;
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            border: 2px solid #e2e8f0;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            cursor: pointer;
        }}
        
        .card-img-wrapper {{
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 1rem;
            height: 200px;
        }}
        .card-img-wrapper img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}

        /* --- LOGIN PAGE: SPLIT CARDS --- */
        [data-testid="column"]:nth-of-type(2) > div {{
            border-radius: 20px !important;
            height: 100%;
            min-height: 550px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        }}
        
        [data-testid="column"]:nth-of-type(3) > div {{
            background-color: white !important;
            border-radius: 20px !important;
            padding: 3rem !important;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            height: 100%;
            min-height: 550px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        /* --- DASHBOARD STYLES --- */
        .hero-banner {{
            position: sticky;
            top: 3.75rem; 
            z-index: 999;
            background: linear-gradient(120deg, #064e3b 0%, #10b981 100%);
            /* Reduced padding: 1rem top/bottom, 2rem left/right */
            padding: 1rem 2rem; 
            border-radius: 16px;
            color: white;
            /* Reduced margin to bring content below it closer */
            margin-bottom: 1.5rem; 
            box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.3);
            backdrop-filter: blur(10px);
        }}
        
        .css-card {{
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
            border: 1px solid #e2e8f0;
        }}
        
        .metric-container {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #0f172a;
        }}

        .stButton > button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            width: 100%;
            transition: all 0.2s;
            box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
        }}
        .stButton > button:hover {{
            background-color: #059669;
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);

        }}

        button[kind="secondary"] {{
            background-color: transparent;
            color: #1e293b;
            border: 1px solid #e2e8f0;
            box-shadow: none;
        }}
        button[kind="secondary"]:hover {{
            border-color: {PRIMARY_COLOR};
            color: {PRIMARY_COLOR};
            background-color: #f0fdf4;
        }}
        
        .stTextInput > div > div > input {{
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #e2e8f0;
        }}
        .stTextInput > div > div > input:focus {{
            border-color: {PRIMARY_COLOR};
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
        }}
        /* Position the button to overlap the card */
        .stButton > button[key^="btn_"] {{
            height: 340px;          /* Keep this the same as your card height */
            margin-top: 10px;     /* CHANGE THIS: Moving it from -360px to -330px creates a 30px gap */
            background-color: transparent !important;
            border: none !important;
            color: transparent !important;
            z-index: 10;
            transition: all 0.3s ease;
      }}

        /* Add a subtle highlight to the card when the invisible button is hovered */
        .stButton > button[key^="btn_"]:hover {{
            background-color: rgba(16, 185, 129, 0.05) !important;
            cursor: pointer;
        }}
        
        
        </style>
    """, unsafe_allow_html=True)

load_css()

# ================= MODEL LOADER =================
@st.cache_resource
def load_models():
    models = {"Rice": None, "Pulses": None}
    
    if os.path.exists("models/pulses_model.h5"):
        models["Pulses"] = tf.keras.models.load_model("models/pulses_model.h5")
    
    if os.path.exists("models/rice_model.h5"):
        models["Rice"] = tf.keras.models.load_model("models/rice_model.h5")

    return models

loaded_models = load_models()

# ================= HELPERS =================
def get_class_names(choice):
    filename = "static/rice_classes.txt" if choice == "Rice" else "static/pulses_classes.txt"
    try:
        with open(filename, "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        return []

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ================= SESSION MANAGEMENT =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "crop_choice" not in st.session_state:
    st.session_state.crop_choice = None

def login_user(username):
    st.session_state.logged_in = True
    st.session_state.username = username
    st.session_state.page = "dashboard"

def logout_user():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.crop_choice = None
    st.session_state.page = "dashboard"

# ================= NAVIGATION & SIDEBAR =================
def render_sidebar():
    with st.sidebar:
        st.markdown("""<style>.stApp { overflow: auto !important; }</style>""", unsafe_allow_html=True)
        st.title("LeafSense AI")
        st.write(f"Logged in as: **{st.session_state.username}**")
        st.markdown("---")
        
        if st.button("Dashboard", use_container_width=True, type="primary" if st.session_state.page == "dashboard" else "secondary"):
            st.session_state.page = "dashboard"
            st.rerun()
            
        if st.button("History", use_container_width=True, type="primary" if st.session_state.page == "history" else "secondary"):
            st.session_state.page = "history"
            st.rerun()

        st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
        if st.button("Log Out", use_container_width=True):
            logout_user()
            st.rerun()

# ================= PAGE 1: LOGIN =================
def login_page():
    st.markdown("""<style>::-webkit-scrollbar { display: none; } .stApp { overflow: hidden !important; } .block-container { padding-top: 5rem !important; }</style>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col_hero, col_form, col4 = st.columns([0.2, 1, 1, 0.2], gap="large")

    with col_hero:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(5, 150, 105, 0.9) 0%, rgba(4, 120, 87, 0.95) 100%), url('https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2826&auto=format&fit=crop'); background-size: cover; background-position: center; border-radius: 20px; padding: 3rem; height: 100%; min-height: 500px; display: flex; flex-direction: column; justify-content: space-between; color: white;">
            <div>
                <div style="background: rgba(255,255,255,0.2); width: 60px; height: 60px; border-radius: 12px; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(5px);"><span style="font-size: 30px;">🌿</span></div>
                <h1 style="color: white !important; margin-top: 20px; font-size: 2.2rem; font-weight: 700;">LeafSense AI</h1>
                <p style="color: #d1fae5; font-size: 1.1rem; margin-top: 10px; line-height: 1.6;">Professional plant disease diagnostics powered by deep learning.</p>
            </div>
            <div style="display: flex; gap: 8px; margin-bottom: 20px;"><div style="background: white; width: 8px; height: 8px; border-radius: 50%;"></div><div style="background: white; width: 8px; height: 8px; border-radius: 50%; opacity: 0.5;"></div><div style="background: white; width: 8px; height: 8px; border-radius: 50%; opacity: 0.5;"></div></div>
        </div>
        """, unsafe_allow_html=True)

    with col_form:
        st.markdown("""<h2 style="color: #1e293b; margin-bottom: 10px;">Welcome Back</h2><p style="color: #64748b; margin-bottom: 30px;">Enter your credentials to access the dashboard.</p>""", unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["Log In", "Create Account"])
        with tab_login:
            u_email = st.text_input("Email", key="login_user")
            u_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Sign In →", use_container_width=True):
                status, response = authenticate_user(u_email, u_pass)
                if status: login_user(u_email); st.rerun()
                else: st.error(response)
        with tab_register:
            new_user = st.text_input("Email", key="reg_user")
            new_pass = st.text_input("New Password", type="password", key="reg_pass")
            if st.button("Create Account", use_container_width=True):
                status, message = register_user(new_user, new_pass)
                if status: st.success(message)
                else: st.error(message)

# ================= PAGE 2: DASHBOARD VIEW =================
def dashboard_view():
    if st.session_state.page == "dashboard":
        rice_base64 = get_base64_img("static/rice_pic.jpg")
        pulses_base64 = get_base64_img("static/pulses_pic.jpeg")
        # Only show the Dashboard hero banner and Selection Cards if NO crop is selected
        if not st.session_state.crop_choice:
            # The Dashboard Banner
            st.markdown(f"""
            <div class="hero-banner">
                <h1 style='text-align: center;'>LeafSense AI</h1>
                        <h6 style='text-align: center;'>Select a crop to analyze its leaf health</h6>
            """, unsafe_allow_html=True)

            col_rice, col_pulses = st.columns(2, gap="large")
            

            with col_rice:
                st.markdown(f"""
                    <div class="selection-card">
                        <div class="card-img-wrapper">
                            <img src="data:image/jpeg;base64,{rice_base64}">
                        </div>
                        <h4 style="margin: 0; color: #1e293b;">Rice Detection</h4>
                        <p style="color: #64748b; font-size: 0.9rem;">Analyze paddy leaf diseases</p>
                    </div>
                """, unsafe_allow_html=True)
                if st.button("Analyze Rice crops", key="btn_rice", use_container_width=True):
                    st.session_state.crop_choice = "Rice"
                    st.rerun()

            with col_pulses:
                st.markdown(f"""
                    <div class="selection-card">
                        <div class="card-img-wrapper">
                            <img src="data:image/jpeg;base64,{pulses_base64}">
                        </div>
                        <h4 style="margin: 0; color: #1e293b;">Pulses Detection</h4>
                        <p style="color: #64748b; font-size: 0.9rem;">Analyze bean and pea diseases</p>
                    </div>
                """, unsafe_allow_html=True)
                if st.button("Analyze Pulses crops", key="btn_pulses", use_container_width=True):
                    st.session_state.crop_choice = "Pulses"
                    st.rerun()
        # This part triggers ONLY after a crop is selected
        else:
            if st.button("← Back to Crop Selection", type="secondary"):
                st.session_state.crop_choice = None
                st.rerun()

            col1, col2 = st.columns([1, 1.4], gap="medium")
            current_model = loaded_models.get(st.session_state.crop_choice)
            class_names = get_class_names(st.session_state.crop_choice)

            with col1:
                st.markdown(f"<h3>Upload {st.session_state.crop_choice} Leaf</h3>", unsafe_allow_html=True)
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True, caption="Source Image")

            with col2:
                st.markdown("<h3>Analysis Results</h3>", unsafe_allow_html=True)
                if uploaded_file and st.button("Analyze Leaf", use_container_width=True):
                    if current_model:
                        with st.spinner(f"Accessing {st.session_state.crop_choice} Neural Engine..."):
                            img = preprocess_image(image)
                            preds = current_model.predict(img)
                            top_indices = np.argsort(preds[0])[-10:][::-1]
                            idx = top_indices[0]
                            accuracy = preds[0][idx] * 100
                            label = class_names[idx] if idx < len(class_names) else "Unknown"

                            if "___" in label:
                                plant, disease = label.split("___")
                            else:
                                plant, disease = st.session_state.crop_choice, label
                        
                            is_healthy = "healthy" in disease.lower()
                            color = PRIMARY_COLOR if is_healthy else "#ef4444"
                            save_scan(st.session_state.username, plant, disease.replace('_', ' '), float(accuracy), "Healthy" if is_healthy else "Infected")

                            st.markdown(f"""
                            <div style="background: {color}15; border: 1px solid {color}40; border-radius: 12px; padding: 1.5rem; display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                                <div>
                                    <div style="color: {color}; font-size: 0.9rem; font-weight: 600; text-transform: uppercase;">Condition</div>
                                    <div style="color: {color}; font-size: 1.8rem; font-weight: 700;">{disease.replace('_', ' ')}</div>
                                    <div style="color: #64748b; font-size: 0.9rem;">Crop: <b>{plant}</b></div>
                                </div>
                                <div style="font-size: 3rem;">{'✅' if is_healthy else '⚠️'}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.metric("Model Confidence", f"{accuracy:.2f}%")
                            st.progress(int(accuracy) / 100)

                            df_top = pd.DataFrame({
                                "Condition": [class_names[i] if i < len(class_names) else f"Idx {i}" for i in top_indices],
                                "Confidence": [preds[0][i] * 100 for i in top_indices]
                            })
                            st.dataframe(df_top, column_config={"Confidence": st.column_config.ProgressColumn("Probability", format="%.2f%%", min_value=0, max_value=100)}, use_container_width=True, hide_index=True)
                    else: st.error(f"{st.session_state.crop_choice} model not loaded.")
                elif not uploaded_file: st.info("Please upload an image to begin.")

# ================= PAGE 3: HISTORY VIEW =================
def history_page():
    if st.session_state.page == "history":
        st.markdown("<h2 style='text-align: center;'>Scan History</h2>", unsafe_allow_html=True)
        history_data = get_user_history(st.session_state.username)
        if not history_data:
            st.info("No scans found.")
        else:
            df = pd.DataFrame(history_data)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Scans", len(df))
            m2.metric("Healthy", len(df[df['status'] == 'Healthy']))
            m3.metric("Infected", len(df[df['status'] == 'Infected']), delta_color="inverse")
            df = df.rename(columns={"date": "Date", "plant": "Crop", "disease": "Diagnosis", "confidence": "Confidence", "status": "Status"})
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ================= APP CONTROLLER =================
if st.session_state.logged_in:
    render_sidebar()
    if st.session_state.page == "dashboard": dashboard_view()
    elif st.session_state.page == "history": history_page()
else:
    login_page()

