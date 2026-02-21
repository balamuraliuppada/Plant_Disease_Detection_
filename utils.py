from firebase_config import auth

from firestore_db import db
from firebase_admin import firestore


from datetime import datetime


# ---------- FIREBASE AUTH ----------

def register_user(email, password):

    try:
        if len(password) < 6:
            return False, "Password should be at least 6 characters"
        auth.create_user_with_email_and_password(email, password)
        return True, "Registration Successful"

    except Exception as e:
        return False, str(e)


def authenticate_user(email, password):

    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return True, user

    except:
        return False, "Invalid email or password"


# ---------- FIRESTORE HISTORY ----------

def save_scan(email, plant, disease, confidence, status):

    user_ref = db.collection("users").document(email)

    history_ref = user_ref.collection("history")

    history_ref.add({
        "date": firestore.SERVER_TIMESTAMP,
        "plant": plant,
        "disease": disease,
        "confidence": confidence,
        "status": status
    })


def get_user_history(email):

    history_ref = (
        db.collection("users")
        .document(email)
        .collection("history")
        .order_by("date", direction=firestore.Query.DESCENDING)
    )

    docs = history_ref.stream()

    history = []

    for doc in docs:

        data = doc.to_dict()

        if data.get("date"):
            data["date"] = data["date"].strftime("%Y-%m-%d %H:%M")

        history.append(data)

    return history
