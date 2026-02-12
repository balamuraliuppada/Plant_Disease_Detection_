import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from utils import register_user, authenticate_user, save_scan, get_user_history

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
        /* STICKY HERO BANNER */
        .hero-banner {{
            position: sticky;
            top: 3.75rem; 
            z-index: 999;
            background: linear-gradient(120deg, #064e3b 0%, #10b981 100%);
            padding: 2rem 3rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
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
        
        /* Metric Box Styling */
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

        /* Button Styling */
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

        /* Secondary Sidebar Buttons */
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
        
        /* Input Fields */
        .stTextInput > div > div > input {{
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #e2e8f0;
        }}
        .stTextInput > div > div > input:focus {{
            border-color: {PRIMARY_COLOR};
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
        }}
        </style>
    """, unsafe_allow_html=True)

load_css()

# ================= MODEL LOADER (STEP 1: Updated) =================
@st.cache_resource
def load_models():
    main_model = None

    if os.path.exists("model_82.h5"):
        main_model = tf.keras.models.load_model("model_82.h5")

    return main_model
model = load_models()

# ================= CLASS NAMES (STEP 2: Added Rice Classes) =================
try:
    with open("class_names3.txt", "r") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    CLASS_NAMES = [] 


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
# Track active page
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

def login_user(username):
    st.session_state.logged_in = True
    st.session_state.username = username
    st.session_state.page = "dashboard"

def logout_user():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.page = "dashboard"

# ================= NAVIGATION & SIDEBAR =================
def render_sidebar():
    with st.sidebar:
        # Enable scrolling for dashboard
        st.markdown("""
            <style>
            .stApp { overflow: auto !important; }
            </style>
        """, unsafe_allow_html=True)
        
        st.title("LeafSense AI")
        st.write(f"Logged in as: **{st.session_state.username}**")
        st.markdown("---")
        
        # Navigation Buttons
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
    st.markdown("""
        <style>
        ::-webkit-scrollbar { display: none; }
        .stApp { overflow: hidden !important; }
        .block-container { padding-top: 5rem !important; }
        </style>
    """, unsafe_allow_html=True)
    

    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col_hero, col_form, col4 = st.columns([0.2, 1, 1, 0.2], gap="large")

    with col_hero:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(5, 150, 105, 0.9) 0%, rgba(4, 120, 87, 0.95) 100%), 
                        url('https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2826&auto=format&fit=crop');
            background-size: cover;
            background-position: center;
            border-radius: 20px;
            padding: 3rem;
            height: 100%;
            min-height: 500px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            color: white;
        ">
            <div>
                <div style="background: rgba(255,255,255,0.2); width: 60px; height: 60px; border-radius: 12px; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(5px);">
                    <span style="font-size: 30px;">🌿</span>
                </div>
                <h1 style="color: white !important; margin-top: 20px; font-size: 2.2rem; font-weight: 700;">LeafSense AI</h1>
                <p style="color: #d1fae5; font-size: 1.1rem; margin-top: 10px; line-height: 1.6;">
                    Professional plant disease diagnostics powered by deep learning.
                </p>
            </div>
            <div>
                <div style="display: flex; gap: 8px; margin-bottom: 20px;">
                    <div style="background: white; width: 8px; height: 8px; border-radius: 50%;"></div>
                    <div style="background: white; width: 8px; height: 8px; border-radius: 50%; opacity: 0.5;"></div>
                    <div style="background: white; width: 8px; height: 8px; border-radius: 50%; opacity: 0.5;"></div>
                </div>

        </div>
        """, unsafe_allow_html=True)

    with col_form:
        st.markdown("""
            <h2 style="color: #1e293b; margin-bottom: 10px;">Welcome Back</h2>
            <p style="color: #64748b; margin-bottom: 30px;">Enter your credentials to access the dashboard.</p>
        """, unsafe_allow_html=True)
        
        tab_login, tab_register = st.tabs(["Log In", "Create Account"])
        
        with tab_login:
            st.write("")
            username = st.text_input("Username", key="login_user", placeholder="Enter Username")
            password = st.text_input("Password", type="password", key="login_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Sign In →", use_container_width=True):
                if authenticate_user(username, password):
                    login_user(username)
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")

        with tab_register:
            st.write("")
            new_user = st.text_input("New Username", key="reg_user")
            new_pass = st.text_input("New Password", type="password", key="reg_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Create Account", use_container_width=True):
                if register_user(new_user, new_pass):
                    st.success("Account created! Please log in.")
                else:
                    st.error("Username already taken")

# ================= PAGE 2: DASHBOARD VIEW =================
def dashboard_view():
    st.markdown("""
    <div class="hero-banner">
        <h1 style="color: white !important; margin: 0;">Dashboard</h1>
        <p style="margin-top: 10px; opacity: 0.9;">Upload a high-resolution leaf image for instant disease classification.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.4], gap="medium")

    # --- LEFT: UPLOAD ---
    with col1:
        st.markdown("<h3 style='text-align: center;'>Upload Image</h3>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, width=400, caption="Source Image")

    # --- RIGHT: RESULTS ---
    with col2:
        st.markdown("<h3 style='text-align: center;'>Analysis Results</h3>", unsafe_allow_html=True)
        
        if uploaded_file:
            if st.button("Analyze Leaf", use_container_width=True):
                if model:
                    with st.spinner("Analyzing neural patterns..."):
                        img = preprocess_image(image)
                        preds_main = model.predict(img)

                        top_k = min(10, len(CLASS_NAMES))
                        top_indices = np.argsort(preds_main[0])[-top_k:][::-1]

                        idx = top_indices[0]
                        accuracy = preds_main[0][idx] * 100
                        label = CLASS_NAMES[idx]

                        if "___" in label:
                            plant, disease = label.split("___")
                        else:
                            plant, disease = "Unknown", label
                    

                        is_healthy = "healthy" in disease.lower()
                        color = "#10b981" if is_healthy else "#ef4444"
                        status = "Healthy" if is_healthy else "Infected"
                        
                        save_scan(st.session_state.username, plant, disease.replace('_', ' '), float(accuracy), status)

                        # --- MAIN RESULT CARD ---
                        st.markdown(f"""
                        <div style="background: {color}15; border: 1px solid {color}40; border-radius: 12px; padding: 1.5rem; display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                            <div>
                                <div style="color: {color}; font-size: 0.9rem; font-weight: 600; text-transform: uppercase;">Detected Condition</div>
                                <div style="color: {color}; font-size: 1.8rem; font-weight: 700;">{disease.replace('_', ' ')}</div>
                                <div style="color: #64748b; font-size: 0.9rem;">Plant: <b>{plant}</b></div>
                            </div>
                            <div style="font-size: 3rem;">{'✅' if is_healthy else '⚠️'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.metric("Model Confidence", f"{accuracy:.2f}%")
                        st.progress(int(accuracy) / 100)

                        # --- TOP 10 PREDICTIONS TABLE ---
                        st.markdown("### Top 10 Probabilities")
                        
                        top_labels = [CLASS_NAMES[i] for i in top_indices]
                        top_probs = [preds_main[0][i] * 100 for i in top_indices]
                        
                        df_top = pd.DataFrame({
                            "Diagnosis": top_labels,
                            "Confidence": top_probs
                        })
                        
                        # Display as a clean dataframe
                        st.dataframe(
                            df_top,
                            column_config={
                                "Diagnosis": st.column_config.TextColumn("Condition"),
                                "Confidence": st.column_config.ProgressColumn(
                                    "Probability",
                                    format="%.2f%%",
                                    min_value=0,
                                    max_value=100,
                                ),
                            },
                            use_container_width=True,
                            hide_index=True
                        )

                else:
                    st.error("Model not loaded.")
        else:
            st.info("Waiting for image upload...")

# ================= PAGE 3: HISTORY VIEW =================
def history_page():

    st.markdown("<h2 style='text-align: center;'>Crop History</h2>", unsafe_allow_html=True)
    st.markdown("Review your past diagnostic scans below.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Load real data using utils function
    history_data = get_user_history(st.session_state.username)

    if not history_data:
        st.info("No history found. Go to the Dashboard to run your first scan!")
    else:
        # Convert to DataFrame
        df = pd.DataFrame(history_data)

        # --- TOP LEVEL METRICS ---
        total_scans = len(df)
        healthy_scans = len(df[df['status'] == 'Healthy'])
        infected_scans = len(df[df['status'] == 'Infected'])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Scans", total_scans)
        m2.metric("Healthy Plants", healthy_scans)
        m3.metric("Infected Plants", infected_scans, delta_color="inverse")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- STYLED DATAFRAME ---
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        
        df = df.rename(columns={
            "date": "Date",
            "plant": "Plant Type", 
            "disease": "Diagnosis",
            "confidence": "Accuracy (%)",
            "status": "Status"
        })

        st.dataframe(
            df,
            column_config={
                "Date": st.column_config.TextColumn("Scan Date", width="medium"),
                "Diagnosis": st.column_config.TextColumn("Diagnosis", width="large"),
                "Accuracy (%)": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Status": st.column_config.TextColumn(
                    "Health Status",
                    width="small"
                ),
            },
            use_container_width=True,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ================= APP CONTROLLER =================
if st.session_state.logged_in:
    # Show Sidebar
    render_sidebar()
    
    # Route to correct page
    if st.session_state.page == "dashboard":
        dashboard_view()
    elif st.session_state.page == "history":
        history_page()
else:
    login_page()