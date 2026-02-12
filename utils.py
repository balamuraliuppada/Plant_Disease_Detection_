import json
import hashlib
import os
from datetime import datetime

USER_FILE = "users.json"
HISTORY_FILE = "history.json"

# --- AUTHENTICATION (Existing) ---
def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    try:
        with open(USER_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = hash_password(password)
    save_users(users)
    return True

def authenticate_user(username, password):
    users = load_users()
    if username not in users:
        return False
    return users[username] == hash_password(password)

# --- HISTORY MANAGEMENT (New) ---
def load_all_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_scan(username, plant, disease, confidence, status):
    history = load_all_history()
    
    if username not in history:
        history[username] = []
        
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "plant": plant,
        "disease": disease,
        "confidence": confidence,
        "status": status
    }
    
    # Add to beginning of list (newest first)
    history[username].insert(0, entry)
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def get_user_history(username):
    history = load_all_history()
    return history.get(username, [])