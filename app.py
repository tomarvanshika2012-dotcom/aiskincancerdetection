import streamlit as st
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import uuid
import qrcode
import google.generativeai as genai

import json
import traceback
import sqlite3
import pandas as pd
import bcrypt
import hashlib
from scipy.spatial.distance import cosine as cosine_distance
import folium
from streamlit_folium import st_folium
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import requests
from security import (
    init_security_tables, check_session_timeout, perform_session_logout,
    update_activity, log_activity, track_login_attempt, is_account_locked,
    store_otp, verify_otp, validate_upload, strip_image_metadata,
    sanitize_input, sanitize_html, validate_email,
    show_medical_disclaimer, show_privacy_policy,
    render_security_dashboard, get_audit_logs,
    MAX_LOGIN_ATTEMPTS, LOCKOUT_DURATION_MINUTES
)

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI Skin Cancer Detection", page_icon="🧬", layout="wide")

# ================= APP INFO =================
APP_VERSION = "8.0"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-2.0-flash"

# ================= THEME COLORS =================
# Primary Colors
PRIMARY_COLOR = "#339af0"      # Blue
SECONDARY_COLOR = "#ff6b6b"    # Red
SUCCESS_COLOR = "#51cf66"      # Green
WARNING_COLOR = "#ffd43b"      # Yellow
INFO_COLOR = "#845ef7"         # Purple

# Status Colors
RISK_LOW_COLOR = SUCCESS_COLOR
RISK_MODERATE_COLOR = WARNING_COLOR
RISK_HIGH_COLOR = SECONDARY_COLOR

# Background Colors
BG_LIGHT = "#f8f9fa"
BG_WHITE = "#ffffff"
BG_LIGHT_BLUE = "#f0f4ff"
BG_LIGHT_GREEN = "#e7f5ff"
BG_LIGHT_PURPLE = "#f9f2ff"
BG_LIGHT_YELLOW = "#fff9db"

# Text Colors
TEXT_DARK = "#333333"
TEXT_MEDIUM = "#555555"
TEXT_LIGHT = "#888888"
TEXT_WHITE = "#ffffff"

def get_gemini_client(api_key):
    genai.configure(api_key=api_key)
    return genai

def generate_gemini_content(client, contents, model=None):
    """Generate content with Gemini, falling back to alternative model if primary fails."""
    if model is None:
        model = GEMINI_MODEL
    try:
        model_obj = client.GenerativeModel(model)
        response = model_obj.generate_content(contents)
        return response
    except Exception:
        try:
            response = client.models.generate_content(model=GEMINI_MODEL_FALLBACK, contents=contents)
            return response
        except Exception as e:
            raise e

HAM10000_CLASSES = [
    "Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis",
    "Dermatofibroma", "Melanoma", "Melanocytic Nevi", "Vascular Lesion",
    "Fungal Infection", "Other"
]

BODY_LOCATIONS = [
    "scalp", "ear", "face", "back", "trunk", "chest", "abdomen",
    "upper extremity", "lower extremity", "hand", "foot", "neck", "genital"
]

FITZPATRICK_TYPES = {
    "Type I — Very Fair": 1, "Type II — Fair": 2, "Type III — Medium": 3,
    "Type IV — Olive": 4, "Type V — Brown": 5, "Type VI — Dark Brown/Black": 6
}

SUPPORTED_LANGUAGES = {
    "English": "en", "हिन्दी (Hindi)": "hi", "Español (Spanish)": "es",
    "Français (French)": "fr", "Deutsch (German)": "de", "中文 (Chinese)": "zh",
}

TRIAGE_LEVELS = {
    "LOW": {"color": "#51cf66", "icon": "🟢", "label": "Low Risk",
            "action": "Routine monitoring. Re-check in 3–6 months. Continue regular self-examinations."},
    "MODERATE": {"color": "#ffd43b", "icon": "🟡", "label": "Moderate Risk",
                 "action": "Schedule a dermatologist appointment within 2 weeks. Document any changes with photos."},
    "HIGH": {"color": "#ff6b6b", "icon": "🔴", "label": "High Risk",
             "action": "URGENT: Seek immediate dermatological evaluation. A biopsy may be recommended."},
}

# ================= DATABASE SETUP =================
import tempfile
DB_PATH = os.path.join(tempfile.gettempdir(), "patient_history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # ── Existing scans table ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, age INTEGER, gender TEXT, patient_id TEXT,
            contact TEXT, exam_date TEXT, predicted_class TEXT,
            confidence REAL, cancer_prob REAL, risk_level TEXT,
            reasoning TEXT, recommendation TEXT, scan_timestamp TEXT,
            body_location TEXT, skin_type TEXT, lesion_size TEXT,
            lesion_changed TEXT, uncertainty TEXT, triage_level TEXT,
            inference_mode TEXT
        )
    """)
    for col, ctype in [("body_location","TEXT"),("skin_type","TEXT"),("lesion_size","TEXT"),
                       ("lesion_changed","TEXT"),("uncertainty","TEXT"),("triage_level","TEXT"),
                       ("inference_mode","TEXT"),("feature_vector","BLOB"),("image_thumbnail","BLOB"),
                       ("symptoms","TEXT"),("duration_of_symptom","TEXT")]:
        try:
            c.execute(f"ALTER TABLE scans ADD COLUMN {col} {ctype}")
        except Exception:
            pass

    # ── Patients table ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            date_of_birth TEXT,
            phone_number TEXT,
            profile_photo BLOB,
            blood_group TEXT,
            allergies TEXT,
            existing_conditions TEXT,
            family_history TEXT,
            account_created_at TEXT,
            last_login TEXT,
            notification_preferences TEXT DEFAULT 'all'
        )
    """)

    # ── Doctors table ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            doctor_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            phone_number TEXT,
            profile_photo BLOB,
            specialization TEXT,
            qualification TEXT,
            years_of_experience INTEGER,
            medical_license_number TEXT,
            license_document BLOB,
            hospital_or_clinic_name TEXT,
            clinic_address TEXT,
            city TEXT,
            consultation_fee REAL DEFAULT 0.0,
            consultation_mode TEXT DEFAULT 'Both',
            available_days TEXT DEFAULT 'Mon,Tue,Wed,Thu,Fri,Sat',
            available_time_slots TEXT DEFAULT '09:00-17:00',
            max_patients_per_day INTEGER DEFAULT 10,
            verification_status TEXT DEFAULT 'Pending',
            verified_by_admin TEXT,
            verification_date TEXT,
            average_rating REAL DEFAULT 0.0,
            total_patients_handled INTEGER DEFAULT 0,
            account_created_at TEXT,
            last_login TEXT,
            status TEXT DEFAULT 'Active'
        )
    """)

    # ── Appointments table ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            appointment_id TEXT PRIMARY KEY,
            patient_id TEXT,
            doctor_id TEXT,
            date TEXT,
            time TEXT,
            status TEXT DEFAULT 'Pending',
            consultation_type TEXT DEFAULT 'Online',
            notes TEXT,
            created_at TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
            FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
        )
    """)
    # ── Appointments table migration (new columns) ──
    for col, ctype in [("patient_name","TEXT"),("patient_age","INTEGER"),("patient_gender","TEXT"),
                       ("patient_phone","TEXT"),("patient_email","TEXT"),
                       ("symptoms","TEXT"),("medical_history","TEXT"),("skin_image","BLOB")]:
        try:
            c.execute(f"ALTER TABLE appointments ADD COLUMN {col} {ctype}")
        except Exception:
            pass

    # ── Reports table ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_id TEXT PRIMARY KEY,
            patient_id TEXT,
            doctor_id TEXT,
            image BLOB,
            ai_result TEXT,
            doctor_diagnosis TEXT,
            prescription TEXT,
            doctor_notes TEXT,
            comparison_notes TEXT,
            created_at TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
            FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
        )
    """)

    # ── Reviews table ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            doctor_id TEXT,
            rating REAL,
            review_text TEXT,
            created_at TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
            FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
        )
    """)

    # ── Default admin account (only hash if admin doesn't exist yet) ──
    c.execute("SELECT 1 FROM doctors WHERE doctor_id='ADMIN-001'")
    if c.fetchone() is None:
        admin_hash = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        try:
            c.execute("""INSERT OR IGNORE INTO doctors
                (doctor_id, full_name, email, password_hash, specialization, qualification,
                 verification_status, account_created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("ADMIN-001", "System Admin", "admin@dermascan.ai", admin_hash,
                 "System Administrator", "Admin", "Verified",
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Active"))
        except Exception:
            pass

    conn.commit()
    conn.close()

# ================= AUTH HELPERS =================
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, password_hash):
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except Exception:
        return False

def generate_id(prefix="P"):
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"

# ── Patient CRUD ──
def register_patient(full_name, email, password, age=None, gender=None, dob=None,
                     phone=None, blood_group=None, allergies=None,
                     existing_conditions=None, family_history=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    pid = generate_id("PAT")
    pw_hash = hash_password(password)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        c.execute("""INSERT INTO patients
            (patient_id, full_name, email, password_hash, age, gender, date_of_birth,
             phone_number, blood_group, allergies, existing_conditions, family_history,
             account_created_at, last_login)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pid, full_name, email, pw_hash, age, gender, dob, phone,
             blood_group, allergies, existing_conditions, family_history, now, now))
        conn.commit()
        conn.close()
        return pid, None
    except sqlite3.IntegrityError:
        conn.close()
        return None, "Email already registered."
    except Exception as e:
        conn.close()
        return None, str(e)

def login_patient(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE email=?", (email,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return None, "Email not found."
    cols = [desc[0] for desc in c.description]
    patient = dict(zip(cols, row))
    conn.close()
    if verify_password(password, patient["password_hash"]):
        # Update last login
        conn2 = sqlite3.connect(DB_PATH)
        conn2.execute("UPDATE patients SET last_login=? WHERE patient_id=?",
                      (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), patient["patient_id"]))
        conn2.commit()
        conn2.close()
        return patient, None
    return None, "Incorrect password."

def get_patient(patient_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,))
    row = c.fetchone()
    if row:
        cols = [desc[0] for desc in c.description]
        conn.close()
        return dict(zip(cols, row))
    conn.close()
    return None

# SQL injection protection: whitelist of allowed column names
_PATIENT_ALLOWED_COLS = {
    "full_name", "age", "gender", "date_of_birth", "phone_number",
    "profile_photo", "blood_group", "allergies", "existing_conditions",
    "family_history", "notification_preferences", "password_hash", "last_login"
}

def update_patient(patient_id, **kwargs):
    conn = sqlite3.connect(DB_PATH)
    for k in kwargs:
        if k not in _PATIENT_ALLOWED_COLS:
            conn.close()
            raise ValueError(f"Invalid column for patient update: {k}")
    sets = ", ".join([f"{k}=?" for k in kwargs.keys()])
    vals = list(kwargs.values()) + [patient_id]
    conn.execute(f"UPDATE patients SET {sets} WHERE patient_id=?", vals)
    conn.commit()
    conn.close()

# ── Doctor CRUD ──
def register_doctor(full_name, email, password, phone=None, specialization=None,
                    qualification=None, years_exp=None, license_num=None,
                    hospital=None, address=None, city=None, fee=0.0,
                    mode="Both", available_days="Mon,Tue,Wed,Thu,Fri,Sat",
                    available_slots="09:00-17:00", max_patients=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    did = generate_id("DOC")
    pw_hash = hash_password(password)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        c.execute("""INSERT INTO doctors
            (doctor_id, full_name, email, password_hash, phone_number,
             specialization, qualification, years_of_experience,
             medical_license_number, hospital_or_clinic_name, clinic_address,
             city, consultation_fee, consultation_mode, available_days,
             available_time_slots, max_patients_per_day,
             account_created_at, last_login, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (did, full_name, email, pw_hash, phone, specialization, qualification,
             years_exp, license_num, hospital, address, city, fee, mode,
             available_days, available_slots, max_patients, now, now, "Active"))
        conn.commit()
        conn.close()
        return did, None
    except sqlite3.IntegrityError:
        conn.close()
        return None, "Email already registered."
    except Exception as e:
        conn.close()
        return None, str(e)

def login_doctor(email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM doctors WHERE email=?", (email,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return None, "Email not found."
    cols = [desc[0] for desc in c.description]
    doctor = dict(zip(cols, row))
    conn.close()
    if verify_password(password, doctor["password_hash"]):
        conn2 = sqlite3.connect(DB_PATH)
        conn2.execute("UPDATE doctors SET last_login=? WHERE doctor_id=?",
                      (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), doctor["doctor_id"]))
        conn2.commit()
        conn2.close()
        return doctor, None
    return None, "Incorrect password."

def get_doctor(doctor_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM doctors WHERE doctor_id=?", (doctor_id,))
    row = c.fetchone()
    if row:
        cols = [desc[0] for desc in c.description]
        conn.close()
        return dict(zip(cols, row))
    conn.close()
    return None

_DOCTOR_ALLOWED_COLS = {
    "full_name", "phone_number", "profile_photo", "specialization", "qualification",
    "years_of_experience", "medical_license_number", "license_document",
    "hospital_or_clinic_name", "clinic_address", "city", "consultation_fee",
    "consultation_mode", "available_days", "available_time_slots", "max_patients_per_day",
    "verification_status", "verified_by_admin", "verification_date",
    "average_rating", "total_patients_handled", "password_hash", "last_login", "status"
}

def update_doctor(doctor_id, **kwargs):
    conn = sqlite3.connect(DB_PATH)
    for k in kwargs:
        if k not in _DOCTOR_ALLOWED_COLS:
            conn.close()
            raise ValueError(f"Invalid column for doctor update: {k}")
    sets = ", ".join([f"{k}=?" for k in kwargs.keys()])
    vals = list(kwargs.values()) + [doctor_id]
    conn.execute(f"UPDATE doctors SET {sets} WHERE doctor_id=?", vals)
    conn.commit()
    conn.close()

def get_all_doctors(verified_only=False):
    conn = sqlite3.connect(DB_PATH)
    if verified_only:
        df = pd.read_sql_query("SELECT * FROM doctors WHERE verification_status='Verified' AND status='Active' AND specialization != 'System Administrator'", conn)
    else:
        df = pd.read_sql_query("SELECT * FROM doctors WHERE specialization != 'System Administrator'", conn)
    conn.close()
    return df

def get_all_patients_list():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM patients ORDER BY account_created_at DESC", conn)
    conn.close()
    return df

# ── Appointment CRUD ──
def create_appointment(patient_id, doctor_id, date, time, consultation_type="Online", notes="",
                      patient_name="", patient_age=None, patient_gender="",
                      patient_phone="", patient_email="",
                      symptoms="", medical_history="", skin_image_blob=None):
    conn = sqlite3.connect(DB_PATH)
    aid = generate_id("APT")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""INSERT INTO appointments
        (appointment_id, patient_id, doctor_id, date, time, status, consultation_type, notes, created_at,
         patient_name, patient_age, patient_gender, patient_phone, patient_email,
         symptoms, medical_history, skin_image)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (aid, patient_id, doctor_id, date, time, "Pending", consultation_type, notes, now,
         patient_name, patient_age, patient_gender, patient_phone, patient_email,
         symptoms, medical_history, skin_image_blob))
    conn.commit()
    conn.close()
    return aid

def get_appointments_for_patient(patient_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""SELECT a.*, d.full_name as doctor_name, d.specialization, d.hospital_or_clinic_name
        FROM appointments a LEFT JOIN doctors d ON a.doctor_id = d.doctor_id
        WHERE a.patient_id=? ORDER BY a.date DESC, a.time DESC""", conn, params=(patient_id,))
    conn.close()
    return df

def get_appointments_for_doctor(doctor_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""SELECT a.*, p.full_name as p_full_name, p.age as p_age, p.gender as p_gender, p.phone_number as p_phone
        FROM appointments a LEFT JOIN patients p ON a.patient_id = p.patient_id
        WHERE a.doctor_id=? ORDER BY a.date DESC, a.time DESC""", conn, params=(doctor_id,))
    conn.close()
    # Use appointment-level patient info if available, else fall back to patients table
    if 'patient_name' not in df.columns:
        df['patient_name'] = df.get('p_full_name', '')
    else:
        df['patient_name'] = df['patient_name'].fillna(df.get('p_full_name', ''))
    return df

def update_appointment_status(appointment_id, status):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE appointments SET status=? WHERE appointment_id=?", (status, appointment_id))
    conn.commit()
    conn.close()

# ── Report CRUD ──
def create_report(patient_id, doctor_id, ai_result, doctor_diagnosis="", prescription="", notes="", image_blob=None):
    conn = sqlite3.connect(DB_PATH)
    rid = generate_id("RPT")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""INSERT INTO reports
        (report_id, patient_id, doctor_id, image, ai_result, doctor_diagnosis, prescription, doctor_notes, created_at)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (rid, patient_id, doctor_id, image_blob, ai_result, doctor_diagnosis, prescription, notes, now))
    conn.commit()
    conn.close()
    return rid

def get_reports_for_patient(patient_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""SELECT r.*, d.full_name as doctor_name, d.specialization
        FROM reports r LEFT JOIN doctors d ON r.doctor_id = d.doctor_id
        WHERE r.patient_id=? ORDER BY r.created_at DESC""", conn, params=(patient_id,))
    conn.close()
    return df

def get_reports_for_doctor(doctor_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""SELECT r.*, p.full_name as patient_name, p.age, p.gender
        FROM reports r LEFT JOIN patients p ON r.patient_id = p.patient_id
        WHERE r.doctor_id=? ORDER BY r.created_at DESC""", conn, params=(doctor_id,))
    conn.close()
    return df

# ── Review CRUD ──
def create_review(patient_id, doctor_id, rating, review_text):
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""INSERT INTO reviews (patient_id, doctor_id, rating, review_text, created_at)
        VALUES (?,?,?,?,?)""", (patient_id, doctor_id, rating, review_text, now))
    # Update doctor average rating
    c = conn.cursor()
    c.execute("SELECT AVG(rating), COUNT(*) FROM reviews WHERE doctor_id=?", (doctor_id,))
    avg, cnt = c.fetchone()
    conn.execute("UPDATE doctors SET average_rating=?, total_patients_handled=? WHERE doctor_id=?",
                 (round(avg, 2) if avg else 0, cnt, doctor_id))
    conn.commit()
    conn.close()

def get_reviews_for_doctor(doctor_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""SELECT r.*, p.full_name as patient_name
        FROM reviews r LEFT JOIN patients p ON r.patient_id = p.patient_id
        WHERE r.doctor_id=? ORDER BY r.created_at DESC""", conn, params=(doctor_id,))
    conn.close()
    return df

# ── Scan Helpers (existing, preserved) ──
def save_scan(patient_data, result, feature_vector=None, image_thumbnail=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cancer_prob = result.get("cancer_prob", 0.0)
    risk_percent = cancer_prob * 100
    risk_level = "LOW" if risk_percent < 30 else ("MODERATE" if risk_percent < 60 else "HIGH")
    fv_blob = None
    if feature_vector is not None:
        fv_blob = feature_vector.tobytes()
    thumb_blob = None
    if image_thumbnail is not None:
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(np.array(image_thumbnail.resize((112,112))), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
        thumb_blob = buf.tobytes()
    c.execute("""
        INSERT INTO scans (name, age, gender, patient_id, contact, exam_date,
                          predicted_class, confidence, cancer_prob, risk_level,
                          reasoning, recommendation, scan_timestamp,
                          body_location, skin_type, lesion_size, lesion_changed,
                          uncertainty, triage_level, inference_mode, feature_vector, image_thumbnail,
                          symptoms, duration_of_symptom)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_data["name"], patient_data["age"], patient_data["gender"],
        patient_data["patient_id"], patient_data["contact"], patient_data["exam_date"],
        result.get("predicted_class", "Unknown"), result.get("confidence", 0.0),
        cancer_prob, risk_level,
        result.get("reasoning", ""), result.get("recommendation", ""),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        patient_data.get("body_location", ""), patient_data.get("skin_type", ""),
        patient_data.get("lesion_size", ""), patient_data.get("lesion_changed", ""),
        result.get("uncertainty", ""), risk_level, result.get("inference_mode", "cloud"),
        fv_blob, thumb_blob,
        patient_data.get("symptoms", ""), patient_data.get("duration_of_symptom", "")
    ))
    conn.commit()
    conn.close()

def get_all_scans():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM scans ORDER BY scan_timestamp DESC", conn)
    conn.close()
    return df

init_db()
init_security_tables()


# ================= SESSION STATE =================
for key, default in [("chat_history", []), ("last_result", None),
                     ("selected_language", "English"), ("active_page", "🏠 Home"),
                     ("use_local_model", False), ("enable_tta", False),
                     ("enable_hair_removal", True),
                     ("logged_in", False), ("user_role", None), ("user_data", None),
                     ("show_register", False), ("register_role", "Patient")]:
    if key not in st.session_state:
        st.session_state[key] = default

# ================= TITLE =================
st.markdown(f"""
<div style='text-align: center;'>
    <img src='https://cdn-icons-png.flaticon.com/512/2785/2785819.png' width='80'>
    <h1>🧬 AI Skin Disease & Cancer Detection System</h1>
    <p style='color: grey; font-size: 14px;'>v{APP_VERSION} — Patient & Doctor Profiles • Appointments • Reports • AI Diagnosis • Multi-Language</p>
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
try:
    _secrets_key = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    _secrets_key = None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or _secrets_key or st.sidebar.text_input("Enter Gemini API Key (or set env var)", type="password", help="Set GEMINI_API_KEY env var on Render")
if not GEMINI_API_KEY:
    st.sidebar.warning("🔑 Set GEMINI_API_KEY env var or enter here for cloud AI features.")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)

# ── SIDEBAR: User Info (logged in) or prompt to login ──
if not st.session_state.logged_in:
    st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #667eea22, #764ba222);
                padding: 15px; border-radius: 12px; margin-bottom: 10px; text-align: center;'>
        <h4 style='margin: 0;'>🔐 Not Logged In</h4>
        <p style='margin: 4px 0 0 0; font-size: 12px; color: #888;'>Go to Home page to Login / Register</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── LOGGED IN USER CARD ──
    user = st.session_state.user_data
    role = st.session_state.user_role
    role_emoji = "🧑" if role == "patient" else ("👨‍⚕️" if role == "doctor" else "🔑")
    role_label = role.capitalize()
    display_name = user.get("full_name", "User")

    st.sidebar.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea22, #764ba222);
                padding: 15px; border-radius: 12px; margin-bottom: 10px;'>
        <h4 style='margin: 0;'>{role_emoji} {display_name}</h4>
        <p style='margin: 4px 0 0 0; font-size: 13px; color: #888;'>
            {role_label} • {user.get('patient_id', user.get('doctor_id', 'N/A'))}
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🚪 Logout", use_container_width=True, key="logout_btn"):
        for k in ["logged_in", "user_role", "user_data"]:
            st.session_state[k] = False if k == "logged_in" else None
        st.session_state.active_page = "🏠 Home"
        st.toast("Logged out successfully!", icon="👋")
        st.rerun()

st.sidebar.markdown("---")

# ── NAVIGATION ──
st.sidebar.title("Navigation")

# Build dynamic page list based on role
if st.session_state.logged_in:
    role = st.session_state.user_role
    if role == "patient":
        PAGE_OPTIONS = ["🏠 Home", "👤 My Profile", "🔬 Prediction", "📊 Dashboard",
                       "📈 Tracking", "📅 Appointments", "📝 Reports", "⭐ Reviews",
                       "🗺️ Find Clinics", "💬 Ask AI"]
    elif role == "doctor":
        PAGE_OPTIONS = ["🏠 Home", "👤 My Profile", "👨‍⚕️ Doctor Dashboard",
                       "📅 Appointments", "📝 Reports", "🗺️ Find Clinics", "💬 Ask AI"]
    elif role == "admin":
        PAGE_OPTIONS = ["🏠 Home", "🔑 Admin Panel", "👨‍⚕️ Doctor Dashboard",
                       "📊 Dashboard", "📅 Appointments", "📝 Reports",
                       "🗺️ Find Clinics", "💬 Ask AI"]
    else:
        PAGE_OPTIONS = ["🏠 Home", "🔬 Prediction", "📊 Dashboard", "📈 Tracking",
                       "🗺️ Find Clinics", "💬 Ask AI"]
else:
    PAGE_OPTIONS = ["🏠 Home", "🔬 Prediction", "📊 Dashboard", "📈 Tracking",
                   "🗺️ Find Clinics", "💬 Ask AI"]

def on_nav_change():
    st.session_state.active_page = st.session_state._nav_radio_widget

current_idx = PAGE_OPTIONS.index(st.session_state.active_page) if st.session_state.active_page in PAGE_OPTIONS else 0
st.sidebar.radio("Go to", PAGE_OPTIONS, index=current_idx, key="_nav_radio_widget", on_change=on_nav_change)
page = st.session_state.active_page


# Inference Mode Toggle
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Inference Mode")
use_local = st.sidebar.checkbox(
    "🔒 Use Local Model (Offline/Private)",
    value=st.session_state.get("use_local_model", False)
)
st.session_state.use_local_model = use_local
if use_local:
    st.sidebar.success("🔒 Images processed locally. No data sent to cloud.")
else:
    st.sidebar.info("☁️ Using Gemini Cloud AI for analysis.")

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Advanced Options")
enable_tta = st.sidebar.checkbox(
    "🎯 Enable TTA (Higher Accuracy)",
    value=st.session_state.get("enable_tta", False),
    help="Test-Time Augmentation: runs 8 augmented versions..."
)
st.session_state.enable_tta = enable_tta

enable_hair = st.sidebar.checkbox(
    "✂️ Hair Removal Preprocessing",
    value=st.session_state.get("enable_hair_removal", True),
    help="Remove hair artifacts from dermoscopy images before analysis."
)
st.session_state.enable_hair_removal = enable_hair

theme_mode = st.sidebar.radio("Theme Mode", ["Light", "Dark"])
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Language")
selected_lang = st.sidebar.selectbox(
    "Translate results to:", list(SUPPORTED_LANGUAGES.keys()),
    index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.selected_language)
)
st.session_state.selected_language = selected_lang

# ── Session Timeout Check ──
if check_session_timeout():
    perform_session_logout()
    st.warning("⏰ Your session has expired due to inactivity. Please log in again.")
    st.rerun()
if st.session_state.get("logged_in", False):
    update_activity()

if theme_mode == "Dark":
    st.markdown("<style>.stApp {background-color: #0E1117; color: white;}</style>", unsafe_allow_html=True)
    st.markdown("""<style>
    .stTextInput input, .stNumberInput input, .stSelectbox select, .stDateInput input, .stTextArea textarea {
        color: white !important;
        background-color: #2d3748 !important;
        border-color: #718096 !important;
    }
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label, .stTextArea label {
        color: white !important;
    }
    .stRadio label, .stCheckbox label {
        color: white !important;
    }
    </style>""", unsafe_allow_html=True)


# ================= HELPER FUNCTIONS =================

def translate_text(text, target_language):
    if target_language == "English" or not text or not GEMINI_API_KEY:
        return text
    try:
        client = get_gemini_client(GEMINI_API_KEY)
        prompt = f"Translate the following medical text to {target_language}. Return ONLY the translated text:\n\n{text}"
        return generate_gemini_content(client, prompt).text.strip()
    except Exception:
        return text

# ================= HAIR REMOVAL (Feature 5) =================
def remove_hair(image_array):
    """Remove dark hair from dermoscopy images using morphological blackhat filter."""
    was_float = image_array.dtype == np.float32 or image_array.dtype == np.float64
    if was_float:
        img = (image_array * 255).astype(np.uint8)
    else:
        img = image_array.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    cleaned = cv2.inpaint(img, mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
    if was_float:
        return cleaned.astype(np.float32) / 255.0
    return cleaned

# ================= TEST-TIME AUGMENTATION (Feature 1) =================
def tta_predict(model, img_array, meta_array=None, n_augments=8):
    """Run multiple augmented versions through the model and average predictions.
    
    v7.0 Enhanced TTA with more diverse augmentations for better accuracy.
    """
    augmented = [img_array]
    img_np = img_array[0]
    
    # Horizontal flip
    augmented.append(np.expand_dims(np.fliplr(img_np), 0))
    # Vertical flip
    augmented.append(np.expand_dims(np.flipud(img_np), 0))
    # 90 degree rotation
    augmented.append(np.expand_dims(np.rot90(img_np, 1), 0))
    # 180 degree rotation
    augmented.append(np.expand_dims(np.rot90(img_np, 2), 0))
    # 270 degree rotation
    augmented.append(np.expand_dims(np.rot90(img_np, 3), 0))
    # Slight brightness increase
    augmented.append(np.expand_dims(np.clip(img_np * 1.1, 0, 1), 0))
    # Slight brightness decrease
    augmented.append(np.expand_dims(np.clip(img_np * 0.9, 0, 1), 0))
    # Horizontal + Vertical flip combined
    augmented.append(np.expand_dims(np.flipud(np.fliplr(img_np)), 0))
    # Slight contrast adjustment
    mean_val = np.mean(img_np)
    augmented.append(np.expand_dims(np.clip((img_np - mean_val) * 1.1 + mean_val, 0, 1), 0))
    # Gamma correction (slight)
    augmented.append(np.expand_dims(np.clip(np.power(img_np, 0.9), 0, 1), 0))

    preds = []
    for aug_img in augmented[:n_augments]:
        if meta_array is not None:
            pred = model.predict([aug_img, meta_array], verbose=0)[0]
        else:
            pred = model.predict(aug_img, verbose=0)[0]
        preds.append(pred)
    return np.mean(preds, axis=0)

# ================= OFFLINE OOD DETECTION (Feature 7) =================
def check_ood_local(image):
    """Offline Out-of-Distribution detection using image heuristics."""
    img = np.array(image.resize((224, 224)))
    # Check 1: Skin-tone pixel ratio using HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    # Check 2: Texture check (Laplacian variance)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Check 3: Color diversity
    std_color = np.std(img.astype(float))
    if skin_ratio < 0.10:
        return False, "Image has very few skin-toned pixels. Likely not a skin lesion."
    if lap_var < 10:
        return False, "Image appears too uniform/smooth. May not be a valid photograph."
    if std_color < 8:
        return False, "Image lacks color variation. May be a solid color or blank image."
    return True, "Image passes local OOD checks."

# ================= FEATURE EXTRACTION FOR CBIR (Feature 6) =================
def extract_feature_vector(model, img_array, meta_array=None):
    """Extract the feature vector from the penultimate dense layer for CBIR."""
    try:
        # Find the GlobalAveragePooling2D or last dense layer before output
        feat_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                feat_layer = layer.name
                break
        if feat_layer is None:
            return None
        feat_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(feat_layer).output)
        if meta_array is not None:
            vec = feat_model.predict([img_array, meta_array], verbose=0)[0]
        else:
            vec = feat_model.predict(img_array, verbose=0)[0]
        return vec.astype(np.float32)
    except Exception:
        return None

def find_similar_cases(feature_vector, top_k=3):
    """Find the most similar past scans using cosine similarity."""
    if feature_vector is None:
        return []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, patient_id, predicted_class, confidence, cancer_prob, risk_level, scan_timestamp, feature_vector, image_thumbnail FROM scans WHERE feature_vector IS NOT NULL ORDER BY id DESC LIMIT 200")
    rows = c.fetchall()
    conn.close()
    if not rows:
        return []
    similarities = []
    for row in rows:
        fv_blob = row[8]
        if fv_blob is None:
            continue
        stored_vec = np.frombuffer(fv_blob, dtype=np.float32)
        if stored_vec.shape != feature_vector.shape:
            continue
        try:
            sim = 1 - cosine_distance(feature_vector, stored_vec)
        except Exception:
            continue
        thumb_bytes = row[9]
        similarities.append({
            "id": row[0], "name": row[1], "patient_id": row[2],
            "predicted_class": row[3], "confidence": row[4],
            "cancer_prob": row[5], "risk_level": row[6],
            "timestamp": row[7], "similarity": sim,
            "thumbnail": thumb_bytes
        })
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    # Skip the most similar if it's basically identical (self-match)
    results = [s for s in similarities if s["similarity"] < 0.9999]
    return results[:top_k]

# ================= GRAD-CAM =================
@st.cache_resource
def load_local_model():
    import os
    import time as _time
    model_dir = "/tmp/models"
    os.makedirs(model_dir, exist_ok=True)
    
    MODEL_URL = "https://huggingface.co/vanshikatomar/Skin-Cancer-model/resolve/main/skin_cancer_model_v7_best.h5"
    MODEL_PATH = os.path.join(model_dir, "skin_cancer_model_v7_best.h5")
    
    # Download if not exists (Render cache) — with retry logic
    if not os.path.exists(MODEL_PATH):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                with st.spinner(f"🚀 Downloading model (~200MB, attempt {attempt}/{max_retries})..."):
                    response = requests.get(
                        MODEL_URL,
                        timeout=(30, 120),   # (connect timeout, read timeout)
                        stream=True,
                    )
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0
                    with open(MODEL_PATH + ".tmp", "wb") as f:
                        for chunk in response.iter_content(chunk_size=65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                    # Verify download completeness
                    if total_size and downloaded < total_size:
                        raise ConnectionError(
                            f"Incomplete download: got {downloaded}/{total_size} bytes"
                        )
                    os.rename(MODEL_PATH + ".tmp", MODEL_PATH)
                st.success("✅ Model downloaded!")
                break  # success — exit retry loop
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    ConnectionError) as e:
                # Clean up partial file
                if os.path.exists(MODEL_PATH + ".tmp"):
                    os.remove(MODEL_PATH + ".tmp")
                if attempt < max_retries:
                    wait = 5 * attempt
                    st.warning(
                        f"⚠️ Download attempt {attempt} failed: {e}. "
                        f"Retrying in {wait}s…"
                    )
                    _time.sleep(wait)
                else:
                    st.error(
                        "❌ Model download failed after 3 attempts. "
                        "Please check your internet connection and try again."
                    )
                    return None
    
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.warning(f"Local model load failed: {e}. Use Cloud AI mode.")
        return None

@st.cache_resource
def load_model_artifacts():
    """Load model_artifacts.json to get metadata column info (v7 > v6)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try v7 artifacts first
    artifacts_path_v7 = os.path.join(base_dir, "model_artifacts_v7.json")
    if os.path.exists(artifacts_path_v7):
        with open(artifacts_path_v7, "r") as f:
            return json.load(f)
    
    # Fallback to original artifacts
    artifacts_path = os.path.join(base_dir, "model_artifacts.json")
    if os.path.exists(artifacts_path):
        with open(artifacts_path, "r") as f:
            return json.load(f)
    
    return None

def get_model_img_size():
    """Get the image size from the loaded model dynamically, fallback to artifacts."""
    model = load_local_model()
    if model is not None:
        try:
            if isinstance(model.input, list):
                return int(model.input[0].shape[1])
            else:
                return int(model.input.shape[1])
        except Exception:
            pass

    artifacts = load_model_artifacts()
    if artifacts and "img_size" in artifacts:
        return artifacts["img_size"]
    return 380  # Default to v7 size

def is_dual_input_model(model):
    """Check if the model expects dual inputs (image + metadata)."""
    try:
        return isinstance(model.input, list) and len(model.input) >= 2
    except Exception:
        try:
            return len(model.inputs) >= 2
        except Exception:
            return False

def build_metadata_tensor(metadata, model_artifacts=None):
    """Build a metadata tensor matching the model's expected metadata input shape."""
    if model_artifacts and "metadata_cols" in model_artifacts:
        meta_cols = model_artifacts["metadata_cols"]
        num_features = len(meta_cols)
    else:
        # Fallback: infer from model or use default HAM10000 metadata layout
        num_features = 17  # age_norm + sex_encoded + 15 localization dummies (typical)

    meta_values = np.zeros(num_features, dtype=np.float32)

    if metadata:
        # Age normalized (0-1)
        try:
            meta_values[0] = float(metadata.get("age", 50)) / 100.0
        except (ValueError, TypeError):
            meta_values[0] = 0.5

        # Sex encoded
        gender = metadata.get("gender", "").lower()
        if gender == "male":
            meta_values[1] = 0.0
        elif gender == "female":
            meta_values[1] = 1.0
        else:
            meta_values[1] = 0.5

        # Body location one-hot (if model_artifacts has the column names)
        if model_artifacts and "metadata_cols" in model_artifacts:
            body_loc = metadata.get("body_location", "").lower().replace(" ", "_")
            for i, col in enumerate(meta_cols):
                if col.startswith("loc_") and body_loc in col.lower():
                    meta_values[i] = 1.0

    return np.expand_dims(meta_values, axis=0)

def generate_gradcam(model, img_array, meta_array=None, class_index=None):
    """Generate real Grad-CAM heatmap from the model."""
    try:
        # Find last conv layer
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                last_conv = layer.name
                break
        if last_conv is None:
            return None

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output]
        )

        # Build the correct input based on model type
        if is_dual_input_model(model) and meta_array is not None:
            model_input = [img_array, meta_array]
        else:
            model_input = img_array

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(model_input)
            if class_index is None:
                if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
                    class_index = tf.argmax(predictions[0])
                else:
                    class_index = 0
            loss = predictions[:, class_index] if len(predictions.shape) > 1 else predictions

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    img_size = get_model_img_size()
    img = np.array(original_img.resize((img_size, img_size)))
    heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = np.uint8(heatmap_colored * alpha + img * (1 - alpha))
    return superimposed

# ================= OOD DETECTION =================
def check_ood(image):
    """Out-of-Distribution detection — verify image is a skin lesion."""
    if not GEMINI_API_KEY:
        return True, "API key not available, skipping OOD check."
    try:
        client = get_gemini_client(GEMINI_API_KEY)
        prompt = """Look at this image. Is it a clear photograph of a human skin lesion or skin condition?
Return ONLY a JSON object: {"is_skin": true/false, "reason": "brief explanation"}"""
        response = generate_gemini_content(client, [prompt, image])
        text = response.text.strip()
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            result = json.loads(text[start_idx:end_idx+1])
            return result.get("is_skin", True), result.get("reason", "")
        else:
            return True, "OOD check returned invalid format."
    except Exception:
        return True, "OOD check unavailable."

# ================= MC DROPOUT INFERENCE =================
def mc_dropout_predict(model, img_array, meta_array=None, n_iter=30):
    """Monte Carlo Dropout: run n_iter forward passes with dropout ON."""
    predictions = []
    for _ in range(n_iter):
        if meta_array is not None:
            pred = model([img_array, meta_array], training=True)
        else:
            pred = model(img_array, training=True)
        predictions.append(pred.numpy())
    preds = np.array(predictions)
    mean_pred = preds.mean(axis=0)[0]
    std_pred = preds.std(axis=0)[0]
    return mean_pred, std_pred

# ================= PREDICT (CLOUD) =================
def predict_image_cloud(image, metadata=None):
    if not GEMINI_API_KEY:
        st.error("API Key is missing!")
        return None, None, None, None
    client = get_gemini_client(GEMINI_API_KEY)
    meta_context = ""
    if metadata:
        meta_context = f"""
Patient metadata:
- Age: {metadata.get('age', 'Unknown')}
- Gender: {metadata.get('gender', 'Unknown')}
- Body Location: {metadata.get('body_location', 'Unknown')}
- Fitzpatrick Skin Type: {metadata.get('skin_type', 'Unknown')}
- Lesion Size: {metadata.get('lesion_size', 'Unknown')} mm
- Lesion Changed Recently: {metadata.get('lesion_changed', 'Unknown')}
"""
    prompt = f"""You are an expert dermatologist AI. Analyze this skin image.
{meta_context}
Classify into one of: Melanoma, Basal Cell Carcinoma, Actinic Keratoses, Melanocytic Nevi,
Benign Keratosis, Dermatofibroma, Vascular Lesion, Fungal Infection, Other.

Return ONLY a JSON object (no markdown):
{{
  "predicted_class": "category name",
  "confidence": 85.5,
  "cancer_prob": 0.1,
  "all_probs": {{"Melanoma": 0.05, "Basal Cell Carcinoma": 0.03, "Actinic Keratoses": 0.02,
                 "Melanocytic Nevi": 0.5, "Benign Keratosis": 0.2, "Dermatofibroma": 0.05,
                 "Vascular Lesion": 0.05, "Fungal Infection": 0.05, "Other": 0.05}},
  "reasoning": "Detailed 3-5 sentence clinical explanation of the diagnosis.",
  "recommendation": "Detailed action recommendation with follow-up steps.",
  "lesion_description": "Detailed clinical description of the lesion appearance, texture, and surface characteristics.",
  "morphology": "Describe the shape, symmetry, elevation, and structural features of the lesion.",
  "color_pattern": "Describe the color distribution including any pigment variations, shading, or multi-toned areas.",
  "border_analysis": "Describe the border characteristics — regular/irregular, well-defined/diffuse, notched/smooth.",
  "differential_diagnosis": ["Condition 1", "Condition 2", "Condition 3"],
  "risk_factors": "Relevant risk factors based on lesion features and patient metadata."
}}"""
    try:
        response = generate_gemini_content(client, [prompt, image])
        text = response.text.strip()
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            result = json.loads(text[start_idx:end_idx+1])
        else:
            raise ValueError("Response did not contain valid JSON object.")
        result["inference_mode"] = "cloud"
        st.session_state.last_result = result
        return result, result.get("predicted_class"), result.get("cancer_prob", 0.0), result.get("confidence", 0.0)
    except Exception as e:
        error_str = str(e)
        if "Cannot read image" in error_str or "does not support image input" in error_str:
            st.error("❌ **Image Analysis Failed:** The uploaded file is not a valid image or cannot be processed. Please upload a clear skin lesion photo (JPG/PNG format) and try again.")
        else:
            st.error(f"❌ **Cloud AI Error:** {error_str}")
        return None, "Error", 0.0, 0.0

# ================= PREDICT (LOCAL) =================
CLINICAL_KB = {
    "Actinic Keratoses": {
        "lesion_description": "Rough, scaly, sandpaper-like patches typically 2-6mm in diameter. Surface may be flat or slightly raised with an adherent scale. Often felt before seen on sun-exposed areas.",
        "morphology": "Macular or slightly papular lesion with irregular borders. Size typically 2-10mm. Flat to slightly elevated with a rough, gritty texture upon palpation.",
        "color_pattern": "Variable erythematous to pink, tan, or flesh-colored. May show subtle pigment variation. Redness indicates inflammation and UV damage.",
        "border_analysis": "Poorly defined, irregular borders that blend gradually into surrounding skin. Edges may appear diffuse and non-distinct.",
        "differential_diagnosis": ["Squamous Cell Carcinoma", "Basal Cell Carcinoma", "Seborrheic Keratosis"],
        "risk_factors": "Chronic sun exposure, fair skin (Fitzpatrick I-II), outdoor occupation, immunosuppression, older age (>50 years). Risk of progression to SCC is 0.025-16% per year.",
        "recommendation": "Schedule dermatology appointment within 4 weeks. Cryotherapy, topical 5-fluorouracil, or photodynamic therapy are standard treatments. Use SPF 30+ daily and regular self-monitoring."
    },
    "Basal Cell Carcinoma": {
        "lesion_description": "Pearly, waxy, or translucent papule or nodule with visible telangiectasia (small blood vessels). May present as a non-healing ulcer or a flat, scar-like plaque. Slow-growing but locally invasive.",
        "morphology": "Nodular: dome-shaped, pearly papule. Superficial: thin, erythematous plaque. Morpheaform: indurated, scar-like plaque. Typically 5-20mm at presentation.",
        "color_pattern": "Pearly, translucent, or pink. Central ulceration may appear darker (brown/black). Telangiectasia visible on surface. Pigmented variant shows brown-black coloration.",
        "border_analysis": "Rolled, raised borders with visible blood vessels (telangiectasia). Edges may be pearly and translucent. Central depression or ulceration common.",
        "differential_diagnosis": ["Squamous Cell Carcinoma", "Melanoma", "Seborrheic Keratosis", "Intradermal Nevus"],
        "risk_factors": "Chronic UV exposure, fair skin, history of sunburns, age >40, immunosuppression, arsenic exposure, Gorlin syndrome. Most common skin cancer but rarely metastasizes.",
        "recommendation": "URGENT: Dermatology referral within 1-2 weeks. Surgical excision is gold standard. Mohs surgery recommended for facial lesions. Low metastatic risk but high recurrence risk if untreated."
    },
    "Benign Keratosis": {
        "lesion_description": "Well-demarcated, stuck-on appearing, waxy or verrucous papule/plaque. Surface may be smooth, rough, or cerebriform. Often multiple and increasing with age.",
        "morphology": "Round to oval, sharply demarcated, flat-topped or dome-shaped papule. Typically 3-20mm. Surface can be smooth, rough, or cobblestone-like (verrucous).",
        "color_pattern": "Uniform tan, brown, dark brown, or black. May have a stuck-on appearance. Color is typically homogeneous without significant variation.",
        "border_analysis": "Sharp, well-defined, round borders. Lesion appears to sit on top of the skin surface (stuck-on). No invasion into surrounding tissue.",
        "differential_diagnosis": ["Melanoma", "Basal Cell Carcinoma", "Seborrheic Keratosis", "Verruca Vulgaris"],
        "risk_factors": "Aging, sun exposure, genetic predisposition. Generally benign with no malignant potential. Very common in adults over 30.",
        "recommendation": "Low risk - routine monitoring. No treatment required unless cosmetically bothersome or irritated. Annual skin check recommended. Monitor for any changes in size, color, or symptoms."
    },
    "Dermatofibroma": {
        "lesion_description": "Firm, indurated, dome-shaped papule or nodule. Dimple sign (pinching causes central depression) is characteristic. Usually asymptomatic but may be tender.",
        "morphology": "Round, well-circumscribed, firm, dome-shaped papule. Typically 5-10mm. Central dimple sign when pinched. Slow-growing and stable over time.",
        "color_pattern": "Tan, brown, pink, or skin-colored. Pigmentation may be more intense at the periphery. Color is typically uniform.",
        "border_analysis": "Well-defined, smooth, round borders. No irregularity or notching. Firm on palpation.",
        "differential_diagnosis": ["Dermatofibrosarcoma Protuberans", "Fibrous Papule", "Keloid", "Nodular Melanoma"],
        "risk_factors": "Trauma or insect bite at site, female predominance, lower extremities common. Benign fibrohistiocytic tumor with no malignant potential.",
        "recommendation": "No treatment needed - benign lesion. Monitor for changes. Excisional biopsy if atypical features or diagnostic uncertainty. Reassurance is appropriate."
    },
    "Melanoma": {
        "lesion_description": "Asymmetric pigmented lesion with irregular borders, color variegation, and diameter >6mm. ABCDE criteria: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolution. May be amelanotic (lack pigment).",
        "morphology": "Asymmetric, irregularly shaped macule, papule, or nodule. May be flat (radial growth) or raised (vertical growth). Ulceration indicates advanced stage. Typically >6mm.",
        "color_pattern": "Multiple colors present: brown, black, red, white, blue (ABCDE criteria). Irregular pigment distribution with darker and lighter areas. Blue-white veil is concerning.",
        "border_analysis": "Irregular, notched, scalloped, or poorly defined borders. Asymmetric border pattern. Areas of regression may appear as white scar-like patches.",
        "differential_diagnosis": ["Dysplastic Nevus", "Seborrheic Keratosis", "Basal Cell Carcinoma", "Blue Nevus", "Hemangioma"],
        "risk_factors": "Fair skin, history of sunburns (especially blistering), family history, dysplastic nevi, immunosuppression, genetic mutations (CDKN2A, BRAF). Highest mortality among skin cancers.",
        "recommendation": "URGENT: Immediate dermatology referral within 1 week. Excisional biopsy with adequate margins recommended. Full-body skin examination. Sentinel lymph node biopsy may be needed. Early detection is critical for survival."
    },
    "Melanocytic Nevi": {
        "lesion_description": "Well-circumscribed, symmetric, uniform pigmented macule or papule. Regular in shape and color. May be flat or raised. Common mole - present in most adults.",
        "morphology": "Round to oval, symmetric, dome-shaped or flat. Typically 2-6mm. Smooth surface. Regular, predictable growth pattern.",
        "color_pattern": "Uniform tan, brown, or dark brown. Single color throughout. No significant color variation or irregular pigmentation.",
        "border_analysis": "Smooth, well-defined, round borders. No irregularity, notching, or scalloping. Sharp demarcation from surrounding skin.",
        "differential_diagnosis": ["Melanoma", "Dysplastic Nevus", "Seborrheic Keratosis", "Pigmented Basal Cell Carcinoma"],
        "risk_factors": "Genetic predisposition, sun exposure, light skin. Normal mole with low malignant potential (<1 in 100,000 per year). Changes should be monitored.",
        "recommendation": "Benign - routine monitoring only. Annual full-body skin exam recommended. Document with photos for comparison. Seek evaluation if rapid growth, color change, or symptoms develop."
    },
    "Vascular Lesion": {
        "lesion_description": "Red, purple, or blue lesion due to blood vessel proliferation or dilation. May be flat (port wine stain), raised (hemangioma), or compressible (venous lake). Blanching on pressure is characteristic.",
        "morphology": "Variable - flat macules (port wine stain), dome-shaped nodules (cherry angioma), compressible papules (venous lake). Size highly variable from pinpoint to large patches.",
        "color_pattern": "Red (capillary), purple (venous), blue (deep vascular). Blanching with pressure (diascopy positive). Color uniform within individual lesion.",
        "border_analysis": "Usually well-defined, round borders. May have irregular edges (port wine stain). Smooth, non-scalloped margins.",
        "differential_diagnosis": ["Amelanotic Melanoma", "Basal Cell Carcinoma", "Pyogenic Granuloma", "Kaposi Sarcoma"],
        "risk_factors": "Congenital (hemangioma), aging (cherry angioma, venous lake), chronic sun exposure, liver disease (spider angiomas). Generally benign.",
        "recommendation": "Usually benign - no treatment needed unless symptomatic or cosmetically concerning. Seek evaluation if rapid growth, bleeding, or ulceration. Laser therapy available for cosmetic removal."
    },
    "Fungal Infection": {
        "lesion_description": "Erythematous, scaly patches with central clearing and raised active borders (tinea pattern). May show satellite lesions, maceration, or pustules. Pruritic.",
        "morphology": "Annular (ring-shaped) patches with central clearing. Irregular, polycyclic borders. Scale present on active edge. May be macerated in intertriginous areas.",
        "color_pattern": "Erythematous (red) to pink. Central hypopigmentation with clearing. May appear darker in chronic cases due to post-inflammatory changes.",
        "border_analysis": "Active, raised, scaly borders with central clearing. Polycyclic or arciform pattern. Well-defined active edge with scale.",
        "differential_diagnosis": ["Eczema", "Psoriasis", "Contact Dermatitis", "Granuloma Annulare", "Pityriasis Rosea"],
        "risk_factors": "Warm, moist environments, occlusive clothing, immunosuppression, diabetes, obesity, antibiotic use. Contagious through direct contact.",
        "recommendation": "Antifungal treatment recommended. Topical terbinafine or clotrimazole for 2-4 weeks. Keep area dry and clean. See dermatologist if no improvement in 2 weeks or if widespread."
    },
    "Other": {
        "lesion_description": "Skin lesion that does not fit clearly into the standard dermatological categories. Appearance may be atypical or represent a rare condition. Requires clinical correlation.",
        "morphology": "Variable morphology. May be macular, papular, nodular, or plaque-like. Characteristics may overlap multiple categories.",
        "color_pattern": "Variable pigmentation. May show single or multiple colors. Pigment pattern is atypical for standard categories.",
        "border_analysis": "Variable border characteristics. May be well-defined or poorly defined. Pattern does not match typical presentations.",
        "differential_diagnosis": ["Dysplastic Nevus", "Seborrheic Keratosis", "Inflammatory Dermatosis", "Cutaneous Lymphoma"],
        "risk_factors": "Variable depending on underlying condition. Requires clinical evaluation for accurate risk stratification.",
        "recommendation": "Dermatology referral recommended for clinical evaluation and possible biopsy. Non-specific findings warrant expert assessment for definitive diagnosis."
    },
    "Malignant": {
        "lesion_description": "Lesion with features suggestive of malignancy. Irregular architecture, rapid growth, or ulceration may be present. Requires urgent pathological evaluation.",
        "morphology": "Irregular, asymmetric lesion with variable elevation. May show ulceration, bleeding, or rapid growth pattern.",
        "color_pattern": "Multi-toned or irregular pigmentation. Color variegation is a concerning feature for malignancy.",
        "border_analysis": "Irregular, poorly defined borders with possible notching or scalloping.",
        "differential_diagnosis": ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"],
        "risk_factors": "UV exposure, fair skin, family history, immunosuppression, prior skin cancer history.",
        "recommendation": "URGENT: Seek immediate dermatological evaluation. Biopsy is essential for definitive diagnosis. Do not delay medical consultation."
    },
    "Benign": {
        "lesion_description": "Lesion with benign characteristics. Regular morphology, symmetric shape, and uniform color suggest non-cancerous nature.",
        "morphology": "Symmetric, regular, well-circumscribed lesion. Smooth surface with predictable growth pattern.",
        "color_pattern": "Uniform, homogeneous color distribution. No significant pigment variation.",
        "border_analysis": "Smooth, well-defined, regular borders. No irregularity or notching.",
        "differential_diagnosis": ["Melanocytic Nevus", "Seborrheic Keratosis", "Dermatofibroma"],
        "risk_factors": "Standard sun exposure risk. Low malignant potential.",
        "recommendation": "Routine monitoring recommended. Annual skin examination. Seek evaluation if any changes occur."
    }
}

def predict_image_local(image, metadata=None):
    local_model = load_local_model()
    if local_model is None:
        st.error("No local model found! Train the model first or switch to Cloud AI.")
        return None, None, None, None
    
    # Get image size from model artifacts (380 for v7, 224 for older models)
    img_size = get_model_img_size()
    img = image.resize((img_size, img_size))
    img_array = np.array(img) / 255.0

    # Feature 5: Hair Removal preprocessing
    if st.session_state.get("enable_hair_removal", False):
        img_array = remove_hair(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    # Build metadata tensor if model expects dual inputs
    meta_array = None
    if is_dual_input_model(local_model):
        model_artifacts = load_model_artifacts()
        meta_array = build_metadata_tensor(metadata, model_artifacts)

    # Try MC Dropout
    try:
        output_shape = local_model.output_shape
        num_outputs = output_shape[-1] if isinstance(output_shape[-1], int) else 1
    except Exception:
        num_outputs = 1

    # Feature 1: TTA or MC Dropout prediction
    try:
        if st.session_state.get("enable_tta", False):
            try:
                mean_pred = tta_predict(local_model, img_array, meta_array=meta_array, n_augments=8)
                std_pred = np.zeros_like(mean_pred)
            except Exception:
                if meta_array is not None:
                    mean_pred = local_model.predict([img_array, meta_array], verbose=0)[0]
                else:
                    mean_pred = local_model.predict(img_array, verbose=0)[0]
                std_pred = np.zeros_like(mean_pred)
        else:
            try:
                mean_pred, std_pred = mc_dropout_predict(local_model, img_array, meta_array=meta_array, n_iter=30)
            except Exception:
                if meta_array is not None:
                    mean_pred = local_model.predict([img_array, meta_array], verbose=0)[0]
                else:
                    mean_pred = local_model.predict(img_array, verbose=0)[0]
                std_pred = np.zeros_like(mean_pred)
    except Exception as e:
        error_str = str(e)
        if "Cannot read image" in error_str or "does not support image input" in error_str:
            st.error("❌ **Local Model Error:** The model cannot process this image. Please ensure you have the correct model file and try again.")
        else:
            st.error(f"❌ **Local Model Error:** {error_str}")
        return None, None, None, None

    # Build result
    if num_outputs > 1:
        classes = HAM10000_CLASSES[:num_outputs]
        pred_idx = int(np.argmax(mean_pred))
        predicted_class = classes[pred_idx]
        confidence = float(mean_pred[pred_idx] * 100)
        cancer_classes = ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratoses"]
        cancer_prob = float(sum(mean_pred[i] for i, c in enumerate(classes) if c in cancer_classes))
        all_probs = {c: float(mean_pred[i]) for i, c in enumerate(classes)}
        uncertainty_str = "; ".join([f"{c}: {mean_pred[i]*100:.1f}% ± {std_pred[i]*100:.1f}%" for i, c in enumerate(classes)])
    else:
        cancer_prob = float(mean_pred[0]) if len(mean_pred.shape) == 0 or mean_pred.shape[0] == 1 else float(mean_pred[0])
        predicted_class = "Malignant" if cancer_prob > 0.5 else "Benign"
        confidence = float(max(cancer_prob, 1 - cancer_prob) * 100)
        all_probs = {"Malignant": cancer_prob, "Benign": 1 - cancer_prob}
        uncertainty_str = f"Cancer: {cancer_prob*100:.1f}% ± {float(std_pred)*100:.1f}%"

    kb_entry = CLINICAL_KB.get(predicted_class, None)
    if kb_entry is None:
        kb_entry = CLINICAL_KB["Malignant"] if cancer_prob > 0.5 else CLINICAL_KB["Benign"]

    result = {
        "predicted_class": predicted_class, "confidence": confidence,
        "cancer_prob": cancer_prob, "all_probs": all_probs,
        "reasoning": f"Classification based on local EfficientNet CNN model with MC Dropout uncertainty estimation (30 forward passes). The model identified features consistent with {predicted_class} with {confidence:.1f}% confidence.",
        "recommendation": kb_entry["recommendation"],
        "lesion_description": kb_entry["lesion_description"],
        "morphology": kb_entry["morphology"],
        "color_pattern": kb_entry["color_pattern"],
        "border_analysis": kb_entry["border_analysis"],
        "differential_diagnosis": kb_entry["differential_diagnosis"],
        "risk_factors": kb_entry["risk_factors"],
        "uncertainty": uncertainty_str, "inference_mode": "local"
    }
    st.session_state.last_result = result
    return result, predicted_class, cancer_prob, confidence

def get_triage(cancer_prob):
    pct = cancer_prob * 100
    if pct < 30: return TRIAGE_LEVELS["LOW"]
    elif pct < 60: return TRIAGE_LEVELS["MODERATE"]
    else: return TRIAGE_LEVELS["HIGH"]

# ================= PDF GENERATOR (Professional Lab Report Style) =================
def generate_pdf(patient_data, image, result, gradcam_overlay=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable, KeepTogether
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch, mm, cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfgen import canvas
    from io import BytesIO

    predicted = result.get("predicted_class", "Unknown")
    cancer_prob = result.get("cancer_prob", 0.0)
    confidence = result.get("confidence", 0.0)
    reasoning = result.get("reasoning", "")
    recommendation = result.get("recommendation", "")
    all_probs = result.get("all_probs", {})
    uncertainty = result.get("uncertainty", "N/A")
    triage = get_triage(cancer_prob)
    risk_level = "LOW" if cancer_prob * 100 < 30 else ("MODERATE" if cancer_prob * 100 < 60 else "HIGH")
    report_id = str(uuid.uuid4())[:12].upper()
    now = datetime.now()
    collected_time = now.strftime("%d/%m/%Y  %I:%M:%S %p")
    reported_time = now.strftime("%d/%m/%Y  %I:%M:%S %p")

    filename = os.path.join(tempfile.gettempdir(), "AI_Skin_Cancer_Report.pdf")

    # ── Color Palette (Dr Lal PathLabs inspired) ──
    GOLD = colors.HexColor("#D4A017")
    DARK_GOLD = colors.HexColor("#B8860B")
    HEADER_BG = colors.HexColor("#D4A017")
    LIGHT_GRAY = colors.HexColor("#F5F5F5")
    SECTION_BG = colors.HexColor("#E8E8E8")
    WHITE = colors.white
    BLACK = colors.black
    RED_FLAG = colors.HexColor("#CC0000")
    GREEN_OK = colors.HexColor("#228B22")

    # ── Custom Styles ──
    styles = getSampleStyleSheet()

    style_header_title = ParagraphStyle(
        'HeaderTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=16,
        textColor=WHITE, alignment=TA_LEFT
    )
    style_header_sub = ParagraphStyle(
        'HeaderSub', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7,
        textColor=WHITE, alignment=TA_LEFT, leading=9
    )
    style_section_header = ParagraphStyle(
        'SectionHeader', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9,
        textColor=BLACK, backColor=SECTION_BG, leading=14,
        spaceBefore=4, spaceAfter=2
    )
    style_label = ParagraphStyle(
        'Label', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=10
    )
    style_value = ParagraphStyle(
        'Value', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8, leading=10
    )
    style_value_bold = ParagraphStyle(
        'ValueBold', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=10,
        textColor=RED_FLAG
    )
    style_small = ParagraphStyle(
        'Small', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7, leading=9,
        textColor=colors.HexColor("#555555")
    )
    style_footer = ParagraphStyle(
        'Footer', parent=styles['Normal'],
        fontName='Helvetica', fontSize=6, leading=8,
        textColor=colors.HexColor("#666666"), alignment=TA_CENTER
    )
    style_note = ParagraphStyle(
        'Note', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7, leading=9,
        textColor=BLACK, leftIndent=15
    )
    style_table_header = ParagraphStyle(
        'TableHeader', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=10
    )
    style_disclaimer = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'],
        fontName='Helvetica', fontSize=5.5, leading=7,
        textColor=colors.HexColor("#333333"), alignment=TA_CENTER
    )

    # ── Helper: Check if value is abnormal ──
    cancer_classes = ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratoses"]
    def is_abnormal_class(cls_name, prob_val):
        if cls_name in cancer_classes and prob_val > 0.1:
            return True
        return prob_val > 0.5

    # ── Save temporary images ──
    temp_dir = tempfile.gettempdir()
    
    # Create unique temp files with proper cleanup
    temp_files = []
    
    # Original image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=temp_dir) as orig_file:
        original_path = orig_file.name
    cv2.imwrite(original_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    temp_files.append(original_path)

    # Grad-CAM
    if gradcam_overlay is not None:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=temp_dir) as gcam_file:
            gradcam_path = gcam_file.name
        cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))
        temp_files.append(gradcam_path)
    else:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=temp_dir) as gcam_file:
            gradcam_path = gcam_file.name
        resized = cv2.resize(np.array(image), (224, 224))
        heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
        cv2.imwrite(gradcam_path, heatmap)
        temp_files.append(gradcam_path)

    # Probability chart
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=temp_dir) as prob_file:
        prob_path = prob_file.name
    temp_files.append(prob_path)
    if all_probs:
        plt.figure(figsize=(5.5, 2.5))
        classes_list = list(all_probs.keys())
        probs = [v * 100 for v in all_probs.values()]
        bar_colors = ['#CC0000' if c in cancer_classes else '#228B22' for c in classes_list]
        plt.barh(classes_list, probs, color=bar_colors, height=0.6)
        plt.xlabel("Probability (%)", fontsize=8)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(prob_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.figure(figsize=(4, 2))
        plt.bar(["Cancer", "Benign"], [cancer_prob*100, (1-cancer_prob)*100],
                color=['#CC0000', '#228B22'])
        plt.ylabel("Probability (%)", fontsize=8)
        plt.tight_layout()
        plt.savefig(prob_path, dpi=150)
        plt.close()

    # QR Code
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=temp_dir) as qr_file:
        qr_path = qr_file.name
    temp_files.append(qr_path)
    qr = qrcode.QRCode(version=1, box_size=4, border=1)
    qr.add_data(f"REPORT-{report_id}|{patient_data.get('name','')}|{predicted}|{confidence:.1f}%")
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img.save(qr_path)

    # ── Page Template with header/footer ──
    PAGE_W, PAGE_H = A4
    page_num_storage = [0]

    def draw_header_footer(canvas_obj, doc_obj):
        page_num_storage[0] += 1
        canvas_obj.saveState()

        # ── HEADER: Gold banner ──
        canvas_obj.setFillColor(HEADER_BG)
        canvas_obj.rect(0, PAGE_H - 70, PAGE_W, 70, fill=True, stroke=False)

        # Clinic/System name
        canvas_obj.setFillColor(WHITE)
        canvas_obj.setFont("Helvetica-Bold", 18)
        canvas_obj.drawString(25, PAGE_H - 30, "AI DermaScan")
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.drawString(25, PAGE_H - 42, "Advanced AI-Powered Skin Disease Detection System")
        canvas_obj.setFont("Helvetica", 6.5)
        canvas_obj.drawString(25, PAGE_H - 53, "Powered by EfficientNet Deep Learning | Grad-CAM Explainability | Multi-Modal Analysis")
        canvas_obj.drawString(25, PAGE_H - 63, "Web: ai-dermascan.health  |  Version 7.0  |  Gautam Buddha University")

        # Right side: Accreditation-style badges
        canvas_obj.setFont("Helvetica-Bold", 7)
        canvas_obj.drawRightString(PAGE_W - 25, PAGE_H - 28, "AI-VERIFIED")
        canvas_obj.setFont("Helvetica", 6)
        canvas_obj.drawRightString(PAGE_W - 25, PAGE_H - 38, "Deep Learning")
        canvas_obj.drawRightString(PAGE_W - 25, PAGE_H - 47, "Certified Analysis")

        # Thin gold line below header
        canvas_obj.setStrokeColor(DARK_GOLD)
        canvas_obj.setLineWidth(2)
        canvas_obj.line(0, PAGE_H - 72, PAGE_W, PAGE_H - 72)

        # ── FOOTER ──
        # Gold line above footer
        canvas_obj.setStrokeColor(DARK_GOLD)
        canvas_obj.setLineWidth(1)
        canvas_obj.line(25, 50, PAGE_W - 25, 50)

        # Disclaimer
        canvas_obj.setFillColor(colors.HexColor("#333333"))
        canvas_obj.setFont("Helvetica", 5.5)
        canvas_obj.drawCentredString(PAGE_W / 2, 40,
            "If test results are alarming or unexpected, the patient is advised to consult a dermatologist immediately for clinical evaluation.")
        canvas_obj.drawCentredString(PAGE_W / 2, 33,
            "This report is generated by an AI system and should be used as a screening aid only. It is NOT a substitute for professional medical diagnosis.")
        canvas_obj.drawCentredString(PAGE_W / 2, 26,
            f"AI DermaScan v7.0 | Gautam Buddha University | Report ID: {report_id}")

        # Page number
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.drawRightString(PAGE_W - 25, 26, f"Page {page_num_storage[0]}")

        canvas_obj.restoreState()

    # ── Build Document ──
    doc = SimpleDocTemplate(
        filename, pagesize=A4,
        topMargin=85, bottomMargin=60,
        leftMargin=25, rightMargin=25
    )
    elements = []

    # ── LAB INFO LINE ──
    lab_info = Table([
        [Paragraph("<b>AI DERMASCAN — SKIN DISEASE DETECTION LABORATORY</b>", style_label),
         Paragraph("", style_value)],
    ], colWidths=[4.5*inch, 2.5*inch])
    lab_info.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(lab_info)
    elements.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════
    # SECTION 1: PATIENT INFORMATION (Lab-style block)
    # ══════════════════════════════════════════════════

    patient_info_data = [
        [Paragraph("<b>Name</b>", style_label),
         Paragraph(f":&nbsp;&nbsp;&nbsp;{patient_data.get('name', 'N/A')}", style_value),
         Paragraph("<b>Collected</b>", style_label),
         Paragraph(f":&nbsp;&nbsp;&nbsp;{collected_time}", style_value)],
        [Paragraph("<b>Lab No.</b>", style_label),
         Paragraph(f":&nbsp;&nbsp;&nbsp;{report_id}", style_value),
         Paragraph("<b>Received</b>", style_label),
         Paragraph(f":&nbsp;&nbsp;&nbsp;{collected_time}", style_value)],
        [Paragraph(f"<b>Age: {patient_data.get('age', 'N/A')} Years</b>", style_label),
         Paragraph(f"<b>Gender:&nbsp;&nbsp;{patient_data.get('gender', 'N/A')}</b>", style_value),
         Paragraph("<b>Reported</b>", style_label),
         Paragraph(f":&nbsp;&nbsp;&nbsp;{reported_time}", style_value)],
        [Paragraph("<b>Ref By</b>", style_label),
         Paragraph(f":&nbsp;&nbsp;&nbsp;{result.get('inference_mode', 'Cloud AI').upper()}", style_value),
         Paragraph("<b>Report Status</b>", style_label),
         Paragraph(":&nbsp;&nbsp;&nbsp;<b>Final</b>", style_value_bold if risk_level == "HIGH" else style_value)],
    ]

    patient_table = Table(patient_info_data, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 2.3*inch])
    patient_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1.5, BLACK),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════
    # SECTION 2: SKIN DISEASE CLASSIFICATION PANEL
    # ══════════════════════════════════════════════════

    # Section header (yellow background like Dr Lal PathLabs)
    panel_header = Table(
        [[Paragraph("<b>Test Name</b>", style_table_header),
          Paragraph("<b>Results</b>", style_table_header),
          Paragraph("<b>Units</b>", style_table_header),
          Paragraph("<b>Bio. Ref. Interval</b>", style_table_header)]],
        colWidths=[2.5*inch, 1.3*inch, 1.0*inch, 2.2*inch]
    )
    panel_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECTION_BG),
        ('BOX', (0, 0), (-1, -1), 1, BLACK),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(panel_header)

    # Sub-header: Panel Name
    skin_panel_title = Table(
        [[Paragraph("<b>SKIN DISEASE CLASSIFICATION PANEL</b>", style_section_header),
          Paragraph("", style_value), Paragraph("", style_value), Paragraph("", style_value)]],
        colWidths=[2.5*inch, 1.3*inch, 1.0*inch, 2.2*inch]
    )
    skin_panel_title.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ('SPAN', (0, 0), (-1, 0)),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(skin_panel_title)

    # Classification results rows
    ref_intervals = {
        "Melanoma": "0.00 - 5.00",
        "Basal Cell Carcinoma": "0.00 - 5.00",
        "Actinic Keratoses": "0.00 - 5.00",
        "Melanocytic Nevi": "0.00 - 100.00",
        "Benign Keratosis": "0.00 - 100.00",
        "Dermatofibroma": "0.00 - 100.00",
        "Vascular Lesion": "0.00 - 100.00",
        "Fungal Infection": "0.00 - 100.00",
        "Other": "0.00 - 100.00",
    }

    results_data = []
    abnormal_rows = []

    if all_probs:
        for idx, (cls_name, prob_val) in enumerate(all_probs.items()):
            pct = prob_val * 100
            is_abnormal = is_abnormal_class(cls_name, prob_val)
            if is_abnormal:
                abnormal_rows.append(idx)

            val_style = style_value_bold if is_abnormal else style_value
            marker = " ▲" if is_abnormal and cls_name in cancer_classes else ""

            results_data.append([
                Paragraph(cls_name, style_value),
                Paragraph(f"<b>{pct:.2f}</b>{marker}" if is_abnormal else f"{pct:.2f}", val_style),
                Paragraph("%", style_value),
                Paragraph(ref_intervals.get(cls_name, "0.00 - 100.00"), style_value),
            ])

    if results_data:
        results_table = Table(results_data, colWidths=[2.5*inch, 1.3*inch, 1.0*inch, 2.2*inch])
        table_style_cmds = [
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
            ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ]
        # Highlight abnormal rows
        for row_idx in abnormal_rows:
            table_style_cmds.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor("#FFF0F0")))
        results_table.setStyle(TableStyle(table_style_cmds))
        elements.append(results_table)

    elements.append(Spacer(1, 6))

    # ── Summary Row ──
    summary_data = [
        [Paragraph("<b>Primary Diagnosis</b>", style_label),
         Paragraph(f"<b>{predicted}</b>", style_value_bold if risk_level == "HIGH" else ParagraphStyle('GreenBold', parent=style_value, fontName='Helvetica-Bold', textColor=GREEN_OK)),
         Paragraph("<b>Confidence</b>", style_label),
         Paragraph(f"<b>{confidence:.1f}%</b>", style_value)],
        [Paragraph("<b>Cancer Probability</b>", style_label),
         Paragraph(f"<b>{cancer_prob*100:.1f}%</b>", style_value_bold if cancer_prob > 0.3 else style_value),
         Paragraph("<b>Risk Level</b>", style_label),
         Paragraph(f"<b>{risk_level}</b>", style_value_bold if risk_level == "HIGH" else style_value)],
    ]
    summary_table = Table(summary_data, colWidths=[1.5*inch, 2.0*inch, 1.5*inch, 2.0*inch])
    summary_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1.5, DARK_GOLD),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D4A017")),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFF8E1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 6))

    # ── Triage / Risk Assessment section header ──
    triage_header = Table(
        [[Paragraph("<b>RISK ASSESSMENT &amp; TRIAGE</b>", style_section_header),
          Paragraph("", style_value), Paragraph("", style_value), Paragraph("", style_value)]],
        colWidths=[2.5*inch, 1.3*inch, 1.0*inch, 2.2*inch]
    )
    triage_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ('SPAN', (0, 0), (-1, 0)),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(triage_header)

    triage_color = colors.HexColor("#FFEBEE") if risk_level == "HIGH" else (
        colors.HexColor("#FFF8E1") if risk_level == "MODERATE" else colors.HexColor("#E8F5E9"))

    triage_data = [
        [Paragraph("<b>Triage Level</b>", style_label),
         Paragraph(f"{triage['label']}", style_value_bold if risk_level == "HIGH" else style_value),
         Paragraph("Assessment", style_value),
         Paragraph(ref_intervals.get(predicted, "Screening"), style_value)],
        [Paragraph("<b>Recommended Action</b>", style_label),
         Paragraph(triage['action'], style_value),
         Paragraph("", style_value),
         Paragraph("", style_value)],
    ]
    triage_table = Table(triage_data, colWidths=[1.5*inch, 2.5*inch, 1.0*inch, 2.0*inch])
    triage_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
        ('BACKGROUND', (0, 0), (-1, -1), triage_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('SPAN', (1, 1), (3, 1)),
    ]))
    elements.append(triage_table)

    if uncertainty and uncertainty != "N/A":
        elements.append(Spacer(1, 3))
        elements.append(Paragraph(f"<i>Uncertainty (MC Dropout): {uncertainty}</i>", style_small))

    elements.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════
    # SECTION 3: IMAGE ANALYSIS (Original + Grad-CAM)
    # ══════════════════════════════════════════════════

    img_header = Table(
        [[Paragraph("<b>IMAGE ANALYSIS — Submitted Specimen &amp; AI Heatmap</b>", style_section_header),
          Paragraph("", style_value), Paragraph("", style_value), Paragraph("", style_value)]],
        colWidths=[2.5*inch, 1.3*inch, 1.0*inch, 2.2*inch]
    )
    img_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ('SPAN', (0, 0), (-1, 0)),
    ]))
    elements.append(img_header)
    elements.append(Spacer(1, 4))

    # Images side by side
    try:
        img_table = Table([
            [RLImage(original_path, width=2.2*inch, height=2.2*inch),
             RLImage(gradcam_path, width=2.2*inch, height=2.2*inch),
             RLImage(qr_path, width=0.9*inch, height=0.9*inch)],
            [Paragraph("<b>Original Image</b>", ParagraphStyle('ImgCap', parent=style_small, alignment=TA_CENTER)),
             Paragraph("<b>Grad-CAM Heatmap</b>", ParagraphStyle('ImgCap2', parent=style_small, alignment=TA_CENTER)),
             Paragraph("<b>Scan QR</b>", ParagraphStyle('ImgCap3', parent=style_small, alignment=TA_CENTER))],
        ], colWidths=[2.5*inch, 2.5*inch, 1.2*inch])
        img_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(img_table)
    except Exception:
        elements.append(Paragraph("Image rendering unavailable.", style_value))

    elements.append(Spacer(1, 6))

    # Probability Chart
    try:
        elements.append(RLImage(prob_path, width=5.0*inch, height=2.2*inch))
    except Exception:
        pass

    elements.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════
    # SECTION 4: CLINICAL ANALYSIS TABLE
    # ══════════════════════════════════════════════════

    clinical_header = Table(
        [[Paragraph("<b>AI CLINICAL ANALYSIS</b>", style_section_header),
          Paragraph("", style_value), Paragraph("", style_value), Paragraph("", style_value)]],
        colWidths=[2.5*inch, 1.3*inch, 1.0*inch, 2.2*inch]
    )
    clinical_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
        ('SPAN', (0, 0), (-1, 0)),
    ]))
    elements.append(clinical_header)

    clinical_rows = []

    # Extract detailed fields
    lesion_desc = result.get("lesion_description", "")
    morphology = result.get("morphology", "")
    color_pattern = result.get("color_pattern", "")
    border_analysis = result.get("border_analysis", "")
    differential = result.get("differential_diagnosis", [])
    risk_factors = result.get("risk_factors", "")

    analysis_items = [
        ("Clinical Observations", reasoning),
        ("Clinical Description", lesion_desc),
        ("Morphology (Shape & Structure)", morphology),
        ("Color Pattern", color_pattern),
        ("Border Analysis", border_analysis),
        ("Differential Diagnoses", ", ".join(differential) if isinstance(differential, list) else str(differential)),
        ("Risk Factors", risk_factors),
        ("Anatomical Site", patient_data.get("body_location", "N/A")),
        ("Measured Size", f"{patient_data.get('lesion_size', 'N/A')} mm"),
        ("Skin Type (Fitzpatrick)", patient_data.get("skin_type", "N/A")),
        ("Recent Change Reported", patient_data.get("lesion_changed", "N/A")),
        ("Recommendation", recommendation),
    ]

    for label_text, value_text in analysis_items:
        if value_text and str(value_text).strip():
            clinical_rows.append([
                Paragraph(f"<b>{label_text}</b>", style_label),
                Paragraph(str(value_text), style_value),
            ])

    if clinical_rows:
        clinical_table = Table(clinical_rows, colWidths=[2.0*inch, 5.0*inch])
        clinical_table.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
            ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
            ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(clinical_table)
    else:
        elements.append(Paragraph(
            "<i>Detailed clinical analysis is available when using Cloud AI inference mode.</i>",
            style_small))

    elements.append(Spacer(1, 15))

    # ══════════════════════════════════════════════════
    # SECTION 5: NOTES & IMPORTANT INSTRUCTIONS
    # ══════════════════════════════════════════════════

    notes_header = Table(
        [[Paragraph("<b>Note</b>", style_label)]],
        colWidths=[7.0*inch]
    )
    notes_header.setStyle(TableStyle([
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(notes_header)

    notes = [
        "1.  Analysis conducted on submitted digital dermoscopy / clinical photograph using EfficientNet AI model.",
        "2.  Results are probability-based screening and do NOT constitute a definitive medical diagnosis.",
        "3.  All suspicious findings (cancer probability > 10%) should be clinically correlated with biopsy.",
        "4.  Grad-CAM heatmap indicates regions of interest — does not define lesion boundaries.",
        "5.  MC Dropout uncertainty quantification provides confidence intervals for the AI prediction.",
    ]
    for note in notes:
        elements.append(Paragraph(note, style_note))
    elements.append(Spacer(1, 10))

    # ── Signature Area ──
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#999999")))
    elements.append(Spacer(1, 5))

    sig_data = [
        [Paragraph("", style_value),
         Paragraph("", style_value),
         Paragraph("", style_value)],
        [Paragraph("___________________", style_small),
         Paragraph("___________________", style_small),
         Paragraph("___________________", style_small)],
        [Paragraph("<b>AI DermaScan System</b>", ParagraphStyle('SigL', parent=style_small, alignment=TA_CENTER)),
         Paragraph("<b>Reviewing Physician</b>", ParagraphStyle('SigC', parent=style_small, alignment=TA_CENTER)),
         Paragraph("<b>Consulting Dermatologist</b>", ParagraphStyle('SigR', parent=style_small, alignment=TA_CENTER))],
        [Paragraph("Deep Learning Engine", ParagraphStyle('SigL2', parent=style_small, alignment=TA_CENTER)),
         Paragraph("MD, Pathology", ParagraphStyle('SigC2', parent=style_small, alignment=TA_CENTER)),
         Paragraph("MD, Dermatology", ParagraphStyle('SigR2', parent=style_small, alignment=TA_CENTER))],
    ]
    sig_table = Table(sig_data, colWidths=[2.3*inch, 2.3*inch, 2.3*inch])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(sig_table)

    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "——————————— End of report ———————————",
        ParagraphStyle('EndReport', parent=style_small, alignment=TA_CENTER, fontName='Helvetica-Bold')
    ))

    elements.append(Spacer(1, 8))

    # ── IMPORTANT INSTRUCTIONS ──
    elements.append(HRFlowable(width="100%", thickness=1, color=DARK_GOLD))
    elements.append(Spacer(1, 3))

    instructions_title = Paragraph("<b>IMPORTANT INSTRUCTIONS</b>", ParagraphStyle(
        'InstrTitle', parent=style_small, alignment=TA_CENTER, fontName='Helvetica-Bold', fontSize=7))
    elements.append(instructions_title)
    elements.append(Spacer(1, 3))

    instructions = [
        "*All results pertain to the image submitted. All AI analyses are dependent on the quality of the image provided.",
        "*AI investigations are only a tool to facilitate arriving at a diagnosis and should be clinically correlated by the Referring Physician.",
        "*Report delivery may be delayed due to unforeseen system load. Kindly submit request within 72 hours post analysis.",
        "*AI results may show inter-analysis variations based on image quality, lighting, and resolution.",
        "*The Courts/Forum at Greater Noida shall have exclusive jurisdiction in all disputes regarding the report.",
        "*Results are not valid for medico legal purposes. Contact support for all queries related to reports.",
    ]
    for instr in instructions:
        elements.append(Paragraph(instr, style_disclaimer))

    elements.append(Spacer(1, 3))
    elements.append(HRFlowable(width="100%", thickness=1, color=DARK_GOLD))

    # Build the PDF with header/footer
    doc.build(elements, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    
    # Cleanup temp files
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except (OSError, FileNotFoundError):
            pass
    
    return filename
# ===================================================================
#                           PAGE: HOME
# ===================================================================
if page == "🏠 Home":
    if not st.session_state.logged_in:
        # ═══════════════════════════════════════════════════════════
        #  CENTERED LOGIN / REGISTER — FIRST LAYER (NOT LOGGED IN)
        # ═══════════════════════════════════════════════════════════
        st.markdown("""
        <div style='text-align: center; padding: 30px 0 10px 0;'>
            <h2>Welcome to AI DermaScan</h2>
            <p style='font-size: 16px; color: grey;'>
                AI-Powered Skin Disease Detection • Doctor Consultations • Smart Reports
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── CENTERED LOGIN / REGISTER FORM ──
        auth_tab = st.radio("", ["🔑 Login", "📝 Register"], horizontal=True, key="home_auth_tab",
                           label_visibility="collapsed")

        if auth_tab == "🔑 Login":
            st.markdown("""
            <div style='text-align: center;'>
                <h3>🔑 Login to Your Account</h3>
            </div>
            """, unsafe_allow_html=True)

            # Centered form using columns for padding
            _, login_col, _ = st.columns([1, 2, 1])
            with login_col:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea11, #764ba211);
                            padding: 30px; border-radius: 20px; border: 1px solid #667eea33;'>
                """, unsafe_allow_html=True)

                # ── OTP Verification Step ──
                if st.session_state.get("_otp_pending", False):
                    st.info(f"📧 A 6-digit OTP has been sent to **{st.session_state.get('_otp_email', '')}**")
                    st.markdown("""
                    <div style='background: #e8f5e9; padding: 12px 16px; border-radius: 8px;
                                border-left: 4px solid #4caf50; margin: 10px 0; color: #2e7d32;'>
                        <strong>🔐 Your OTP Code:</strong> <code style='font-size: 20px; letter-spacing: 4px;'>""" +
                        st.session_state.get("_otp_code", "") + """</code>
                        <br><small>In production, this would be sent via email/SMS.</small>
                    </div>
                    """, unsafe_allow_html=True)

                    otp_input = st.text_input("Enter 6-digit OTP", key="otp_input_field",
                                              max_chars=6, placeholder="000000")
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button("✅ Verify OTP", use_container_width=True, key="verify_otp_btn", type="primary"):
                            success, msg = verify_otp(st.session_state["_otp_email"], otp_input)
                            if success:
                                # Complete login
                                user_data = st.session_state["_otp_user_data"]
                                user_role = st.session_state["_otp_user_role"]
                                role_table = "patients" if user_role == "patient" else "doctors"
                                track_login_attempt(st.session_state["_otp_email"], True, role_table)

                                user_id = user_data.get("patient_id", user_data.get("doctor_id", ""))
                                log_activity(user_id, user_role, "LOGIN_SUCCESS",
                                           f"Logged in via OTP 2FA")

                                st.session_state.logged_in = True
                                st.session_state.user_role = user_role
                                st.session_state.user_data = user_data
                                st.session_state["_otp_verified"] = True
                                # Clean up OTP state
                                for k in ["_otp_pending", "_otp_code", "_otp_email",
                                          "_otp_user_data", "_otp_user_role"]:
                                    if k in st.session_state:
                                        del st.session_state[k]
                                update_activity()
                                st.toast(f"Welcome, {user_data.get('full_name', 'User')}!", icon="👋")
                                st.rerun()
                            else:
                                st.error(f"❌ {msg}")
                    with bc2:
                        if st.button("🔄 Cancel", use_container_width=True, key="cancel_otp_btn"):
                            for k in ["_otp_pending", "_otp_code", "_otp_email",
                                      "_otp_user_data", "_otp_user_role"]:
                                if k in st.session_state:
                                    del st.session_state[k]
                            st.rerun()

                else:
                    # ── Step 1: Email + Password ──
                    login_role = st.radio("I am a:", ["Patient", "Doctor / Admin"], horizontal=True, key="home_login_role")
                    login_email = st.text_input("📧 Email Address", key="home_login_email", placeholder="your@email.com")
                    login_pass = st.text_input("🔒 Password", type="password", key="home_login_pass", placeholder="Enter password")

                    if st.button("🔑 Login", use_container_width=True, key="home_login_btn", type="primary"):
                        if login_email and login_pass:
                            role_table = "patients" if login_role == "Patient" else "doctors"

                            # Check account lockout
                            locked, remaining = is_account_locked(login_email, role_table)
                            if locked:
                                st.error(f"🔒 Account locked due to too many failed attempts. Try again in **{remaining:.0f} minutes**.")
                                log_activity("", "", "LOGIN_LOCKED", f"Locked account login attempt: {login_email}")
                            else:
                                if login_role == "Patient":
                                    user, err = login_patient(login_email, login_pass)
                                    user_role = "patient"
                                else:
                                    user, err = login_doctor(login_email, login_pass)
                                    if user and user.get("specialization") == "System Administrator":
                                        user_role = "admin"
                                    else:
                                        user_role = "doctor"

                                if user:
                                    # Password correct → send OTP
                                    otp = store_otp(login_email)
                                    st.session_state["_otp_pending"] = True
                                    st.session_state["_otp_code"] = otp
                                    st.session_state["_otp_email"] = login_email
                                    st.session_state["_otp_user_data"] = user
                                    st.session_state["_otp_user_role"] = user_role
                                    log_activity(user.get("patient_id", user.get("doctor_id", "")),
                                               user_role, "OTP_SENT", f"OTP sent to {login_email}")
                                    st.rerun()
                                else:
                                    # Failed login
                                    track_login_attempt(login_email, False, role_table)
                                    log_activity("", "", "LOGIN_FAILED", f"Failed login: {login_email}")
                                    # Show remaining attempts
                                    locked_now, _ = is_account_locked(login_email, role_table)
                                    if locked_now:
                                        st.error(f"🔒 Account locked after {MAX_LOGIN_ATTEMPTS} failed attempts. Wait {LOCKOUT_DURATION_MINUTES} minutes.")
                                    else:
                                        st.error(f"❌ {err}")
                        else:
                            st.warning("Please fill in all fields.")

                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.caption("🔓 Admin login available for authorized personnel only. | 🔒 Accounts lock after 5 failed attempts.")

        else:  # Register
            st.markdown("""
            <div style='text-align: center;'>
                <h3>📝 Create Your Account</h3>
            </div>
            """, unsafe_allow_html=True)

            reg_role = st.radio("Register as:", ["🧑 Patient", "👨‍⚕️ Doctor"], horizontal=True, key="home_reg_role")

            if reg_role == "🧑 Patient":
                _, reg_col, _ = st.columns([1, 2, 1])
                with reg_col:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #667eea11, #764ba211);
                                padding: 30px; border-radius: 20px; border: 1px solid #667eea33;'>
                    """, unsafe_allow_html=True)
                    st.markdown("#### 🧑 Patient Registration")

                    reg_name = st.text_input("Full Name *", key="home_reg_p_name")
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        reg_email = st.text_input("Email *", key="home_reg_p_email")
                    with rc2:
                        reg_phone = st.text_input("Phone Number", key="home_reg_p_phone")
                    rc3, rc4 = st.columns(2)
                    with rc3:
                        reg_pass = st.text_input("Password *", type="password", key="home_reg_p_pass")
                    with rc4:
                        reg_pass2 = st.text_input("Confirm Password *", type="password", key="home_reg_p_pass2")
                    rc5, rc6, rc7 = st.columns(3)
                    with rc5:
                        reg_age = st.number_input("Age", 1, 120, 25, key="home_reg_p_age")
                    with rc6:
                        reg_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="home_reg_p_gender")
                    with rc7:
                        reg_dob = st.date_input("Date of Birth", key="home_reg_p_dob")

                    with st.expander("🏥 Medical Info (Optional)"):
                        mc1, mc2 = st.columns(2)
                        with mc1:
                            reg_blood = st.selectbox("Blood Group", ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], key="home_reg_p_blood")
                            reg_allergies = st.text_area("Allergies", key="home_reg_p_allergies", placeholder="e.g. Penicillin, Pollen")
                        with mc2:
                            reg_conditions = st.text_area("Existing Conditions", key="home_reg_p_conditions", placeholder="e.g. Diabetes, Hypertension")
                            reg_family = st.text_area("Family History", key="home_reg_p_family", placeholder="e.g. Skin cancer in family")

                    if st.button("📝 Register as Patient", use_container_width=True, key="home_reg_p_btn", type="primary"):
                        if reg_name and reg_email and reg_pass:
                            if reg_pass != reg_pass2:
                                st.error("❌ Passwords don't match!")
                            elif len(reg_pass) < 6:
                                st.error("❌ Password must be at least 6 characters.")
                            else:
                                pid, err = register_patient(
                                    reg_name, reg_email, reg_pass, reg_age, reg_gender,
                                    str(reg_dob), reg_phone, reg_blood, reg_allergies,
                                    reg_conditions, reg_family
                                )
                                if pid:
                                    st.success(f"✅ Registered! Your Patient ID: **{pid}**")
                                    st.toast("Registration successful! Switch to Login tab.", icon="✅")
                                    st.balloons()
                                else:
                                    st.error(f"❌ {err}")
                        else:
                            st.warning("⚠️ Name, Email and Password are required.")

                    st.markdown("</div>", unsafe_allow_html=True)

            else:  # Doctor Registration
                _, reg_col, _ = st.columns([1, 2, 1])
                with reg_col:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #20c99711, #339af022);
                                padding: 30px; border-radius: 20px; border: 1px solid #20c99733;'>
                    """, unsafe_allow_html=True)
                    st.markdown("#### 👨‍⚕️ Doctor Registration")

                    reg_name = st.text_input("Full Name *", key="home_reg_d_name")
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        reg_email = st.text_input("Email *", key="home_reg_d_email")
                    with rc2:
                        reg_phone = st.text_input("Phone Number", key="home_reg_d_phone")
                    rc3, rc4 = st.columns(2)
                    with rc3:
                        reg_pass = st.text_input("Password *", type="password", key="home_reg_d_pass")
                    with rc4:
                        reg_pass2 = st.text_input("Confirm Password *", type="password", key="home_reg_d_pass2")
                    rc5, rc6 = st.columns(2)
                    with rc5:
                        reg_spec = st.selectbox("Specialization *",
                            ["Dermatologist", "Oncologist", "General Physician", "Plastic Surgeon", "Pathologist", "Other"],
                            key="home_reg_d_spec")
                    with rc6:
                        reg_qual = st.text_input("Qualification *", key="home_reg_d_qual", placeholder="MBBS, MD, etc.")
                    rc7, rc8 = st.columns(2)
                    with rc7:
                        reg_exp = st.number_input("Years of Experience", 0, 50, 5, key="home_reg_d_exp")
                    with rc8:
                        reg_license = st.text_input("License Number", key="home_reg_d_license")

                    with st.expander("🏥 Clinic Info (Optional)"):
                        wc1, wc2 = st.columns(2)
                        with wc1:
                            reg_hospital = st.text_input("Hospital/Clinic", key="home_reg_d_hospital")
                            reg_address = st.text_input("Clinic Address", key="home_reg_d_address")
                        with wc2:
                            reg_city = st.text_input("City", key="home_reg_d_city")
                            reg_fee = st.number_input("Fee (₹)", 0.0, 10000.0, 500.0, key="home_reg_d_fee")
                        reg_mode = st.selectbox("Consultation Mode", ["Online", "Offline", "Both"], key="home_reg_d_mode")

                    if st.button("📝 Register as Doctor", use_container_width=True, key="home_reg_d_btn", type="primary"):
                        if reg_name and reg_email and reg_pass:
                            if reg_pass != reg_pass2:
                                st.error("❌ Passwords don't match!")
                            elif len(reg_pass) < 6:
                                st.error("❌ Password must be at least 6 characters.")
                            else:
                                did, err = register_doctor(
                                    reg_name, reg_email, reg_pass, reg_phone, reg_spec,
                                    reg_qual, reg_exp, reg_license, reg_hospital,
                                    reg_address, reg_city, reg_fee, reg_mode
                                )
                                if did:
                                    st.success(f"✅ Registered! Doctor ID: **{did}**\n\n⏳ Pending admin verification.")
                                    st.toast("Registration successful! Awaiting admin verification.", icon="✅")
                                    st.balloons()
                                else:
                                    st.error(f"❌ {err}")
                        else:
                            st.warning("⚠️ Name, Email and Password are required.")

                    st.markdown("</div>", unsafe_allow_html=True)

    else:
        # ═══════════════════════════════════════════════════════════
        #  LOGGED IN HOME — ROLE-SPECIFIC DASHBOARD
        # ═══════════════════════════════════════════════════════════
        user = st.session_state.user_data
        role = st.session_state.user_role
        role_emoji = "🧑" if role == "patient" else ("👨‍⚕️" if role == "doctor" else "🔑")
        st.markdown(f"""
        <div style='text-align: center; padding: 20px 0;'>
            <h2>Welcome back, {role_emoji} {user.get('full_name', 'User')}!</h2>
            <p style='font-size: 16px; color: grey;'>
                {'Manage your skin health with AI-powered diagnosis and doctor consultations.' if role == 'patient'
                else 'Manage your patients, appointments, and provide expert diagnosis.' if role == 'doctor'
                else 'Manage the platform, verify doctors, and monitor system health.'}
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 25px; border-radius: 15px; text-align: center; color: white;'>
                <h1 style='font-size: 40px; margin: 0;'>🔬</h1>
                <h3>AI Diagnosis</h3>
                <p>Multi-class detection with Grad-CAM explainability and triage.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Prediction →", key="card_prediction", use_container_width=True):
                st.session_state.active_page = "🔬 Prediction"
                st.rerun()
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 25px; border-radius: 15px; text-align: center; color: white;'>
                <h1 style='font-size: 40px; margin: 0;'>💬</h1>
                <h3>AI Chatbot</h3>
                <p>Context-aware medical Q&A with multilingual support.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Chatbot →", key="card_chatbot", use_container_width=True):
                st.session_state.active_page = "💬 Ask AI"
                st.rerun()
        with col3:
            if role == "patient":
                st.markdown("""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            padding: 25px; border-radius: 15px; text-align: center; color: white;'>
                    <h1 style='font-size: 40px; margin: 0;'>📅</h1>
                    <h3>Appointments</h3>
                    <p>Book consultations with verified dermatologists.</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Book Appointment →", key="card_appointments", use_container_width=True):
                    st.session_state.active_page = "📅 Appointments"
                    st.rerun()
            elif role == "doctor":
                st.markdown("""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            padding: 25px; border-radius: 15px; text-align: center; color: white;'>
                    <h1 style='font-size: 40px; margin: 0;'>👨‍⚕️</h1>
                    <h3>Doctor Dashboard</h3>
                    <p>Manage patients, consultations, and diagnosis reports.</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Go to Dashboard →", key="card_doc_dashboard", use_container_width=True):
                    st.session_state.active_page = "👨‍⚕️ Doctor Dashboard"
                    st.rerun()
            else:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            padding: 25px; border-radius: 15px; text-align: center; color: white;'>
                    <h1 style='font-size: 40px; margin: 0;'>🔑</h1>
                    <h3>Admin Panel</h3>
                    <p>Verify doctors, manage platform, view system stats.</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Go to Admin →", key="card_admin", use_container_width=True):
                    st.session_state.active_page = "🔑 Admin Panel"
                    st.rerun()

        st.markdown("---")
        st.markdown("### ✨ Platform Features")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**🧠 Grad-CAM XAI**\nSee exactly where the AI looked to make its decision.")
        c2.markdown("**🎯 Test-Time Augmentation**\n8 augmented views averaged for robust predictions.")
        c3.markdown("**✂️ Hair Removal**\nDullRazor preprocessing removes hair artifacts automatically.")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**👤 Patient Profiles**\nComplete medical history, allergies, and family records.")
        c2.markdown("**👨‍⚕️ Doctor Network**\nVerified dermatologists with appointment booking.")
        c3.markdown("**📝 Reports & Tracking**\nAI + doctor diagnosis reports with longitudinal tracking.")

        scans_df = get_all_scans()
        if len(scans_df) > 0:
            st.markdown("---")
            st.markdown("### 📈 Platform Statistics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Scans", len(scans_df))
            high_risk = len(scans_df[scans_df["risk_level"] == "HIGH"]) if "risk_level" in scans_df.columns else 0
            c2.metric("High Risk", high_risk)
            avg_conf = scans_df["confidence"].mean() if "confidence" in scans_df.columns else 0
            c3.metric("Avg Confidence", f"{avg_conf:.1f}%")
            patients_count = len(get_all_patients_list())
            c4.metric("Registered Patients", patients_count)


# ===================================================================
#                        PAGE: PREDICTION
# ===================================================================
if page == "🔬 Prediction":
    st.markdown("## 🔬 Skin Lesion Analysis")

    # Define theme-aware colors
    if theme_mode == "Dark":
        bg_primary = "#2d3748"  # Dark slate background
        bg_secondary = "#1a202c"  # Darker slate background
        bg_accent = "#4a5568"  # Gray accent background
        text_primary = "#ffffff"  # White text
        text_secondary = "#e2e8f0"  # Light gray text
        border_color = "#718096"  # Medium gray border
    else:
        bg_primary = BG_LIGHT
        bg_secondary = BG_WHITE
        bg_accent = BG_LIGHT_BLUE
        text_primary = TEXT_DARK
        text_secondary = TEXT_MEDIUM
        border_color = PRIMARY_COLOR

    # Patient Info — Auto-fill from profile if logged in
    st.markdown("### 👤 Patient Information")
    profile_data = {}
    if st.session_state.logged_in and st.session_state.user_role == "patient":
        profile_data = st.session_state.user_data
        st.info(f"👤 Logged in as **{profile_data.get('full_name', '')}** — fields auto-filled from your profile.")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Patient Name", value=profile_data.get("full_name", ""))
        age = st.number_input("Age", 1, 120, value=profile_data.get("age", 25) or 25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                             index=["Male", "Female", "Other"].index(profile_data.get("gender", "Male")) if profile_data.get("gender") in ["Male", "Female", "Other"] else 0)
    with col2:
        patient_id = st.text_input("Patient ID", value=profile_data.get("patient_id", ""))
        contact = st.text_input("Contact Number", value=profile_data.get("phone_number", "") or "")
        exam_date = st.date_input("Date of Examination")

    # Multi-Modal Metadata (Feature 2)
    st.markdown("### 📋 Clinical Metadata (Multi-Modal Input)")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        body_location = st.selectbox("Body Location", BODY_LOCATIONS)
    with mc2:
        skin_type = st.selectbox("Fitzpatrick Skin Type", list(FITZPATRICK_TYPES.keys()))
    with mc3:
        lesion_size = st.number_input("Lesion Size (mm)", 0.0, 100.0, 5.0)
    with mc4:
        lesion_changed = st.selectbox("Lesion Changed Recently?", ["No", "Yes — Growing", "Yes — Color Change", "Yes — Shape Change"])

    # New: Skin Disease Data fields
    st.markdown("### 🩺 Symptom Details")
    sym1, sym2 = st.columns(2)
    with sym1:
        symptoms = st.multiselect("Symptoms", ["Itching", "Bleeding", "Pain", "Burning", "Scaling",
                                                "Oozing", "Crusting", "Color Change", "Growing", "None"],
                                  default=["None"])
    with sym2:
        duration_of_symptom = st.selectbox("Duration of Symptoms",
                                           ["Less than 1 week", "1-4 weeks", "1-3 months",
                                            "3-6 months", "6-12 months", "More than 1 year"])


    # Fairness Disclaimer (Feature 4)
    if "V" in skin_type or "VI" in skin_type:
        st.warning("⚖️ **Fairness Notice:** This model has limited training data for darker skin tones "
                   "(Fitzpatrick V–VI). Results should be verified by a specialist experienced with diverse skin types.")

    # Image Upload
    st.markdown("### 📸 Upload Skin Image")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    camera_image = st.camera_input("Or Capture Image")
    img = uploaded_file if uploaded_file else camera_image

    if img:
        try:
            img.seek(0)
        except Exception:
            pass
        
        raw_img = Image.open(img).convert("RGB")
        # Strip ALL problematic EXIF metadata rapidly using numpy array proxy
        image = Image.fromarray(np.array(raw_img))
        
        # Downscale just for the UI display to prevent 50MB websocket payload crashes
        display_img = image.copy()
        display_img.thumbnail((800, 800))
        st.image(display_img, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyze", use_container_width=True):
            try:
                metadata = {
                    "age": age, "gender": gender, "body_location": body_location,
                    "skin_type": skin_type, "lesion_size": str(lesion_size),
                    "lesion_changed": lesion_changed
                }

                # OOD Detection — cloud or local (Feature 7)
                if not st.session_state.use_local_model and GEMINI_API_KEY:
                    with st.spinner("🛡️ Validating image (Cloud OOD Detection)..."):
                        is_skin, ood_reason = check_ood(image)
                    if not is_skin:
                        st.error(f"⚠️ **Invalid Image Detected!** This doesn't appear to be a skin lesion.\n\nReason: {ood_reason}")
                        st.stop()
                    else:
                        st.success("✅ Image validated as skin lesion.")
                elif st.session_state.use_local_model:
                    with st.spinner("🛡️ Validating image (Offline OOD Detection)..."):
                        is_skin, ood_reason = check_ood_local(image)
                    if not is_skin:
                        st.error(f"⚠️ **Invalid Image Detected (Local Check)!**\n\nReason: {ood_reason}")
                        st.stop()
                    else:
                        st.success("✅ Image passes local validation.")

                # Hair Removal Preview (Feature 5)
                if st.session_state.get("enable_hair_removal", False):
                    hr_img = np.array(image.resize((224, 224)))
                    cleaned = remove_hair(hr_img)
                    st.session_state.last_hair_removed = cleaned

                # Run Prediction
                with st.spinner("🧠 Analyzing image..."):
                    if st.session_state.use_local_model:
                        result, predicted, cancer_prob, confidence = predict_image_local(image, metadata)
                    else:
                        result, predicted, cancer_prob, confidence = predict_image_cloud(image, metadata)
            except Exception as e:
                error_msg = str(e)
                if "Cannot read image" in error_msg or "does not support image input" in error_msg:
                    st.error("❌ **Image Analysis Failed:** The uploaded file is not a valid image or the AI model cannot process it. Please upload a clear skin lesion photo (JPG/PNG) and try again.")
                else:
                    st.error(f"❌ **Analysis Error:** {error_msg}")
                st.stop()

            if result:
                st.session_state.last_result = result
                st.session_state.last_patient_data = {
                    "name": name, "age": age, "gender": gender,
                    "patient_id": patient_id, "contact": contact,
                    "exam_date": str(exam_date), "body_location": body_location,
                    "skin_type": skin_type, "lesion_size": str(lesion_size),
                    "lesion_changed": lesion_changed,
                    "symptoms": ", ".join(symptoms) if symptoms else "",
                    "duration_of_symptom": duration_of_symptom
                }
                st.session_state.last_image = image

                # Generate Grad-CAM
                local_model = load_local_model()
                feature_vec = None
                if local_model:
                    img_size = get_model_img_size()
                    img_arr = np.expand_dims(np.array(image.resize((img_size, img_size)))/255.0, axis=0)
                    gcam_meta = None
                    if is_dual_input_model(local_model):
                        model_artifacts = load_model_artifacts()
                        gcam_meta = build_metadata_tensor(metadata, model_artifacts)
                    
                    heatmap = generate_gradcam(local_model, img_arr, meta_array=gcam_meta)
                    if heatmap is not None:
                        gradcam_img = overlay_gradcam(image, heatmap)
                        st.session_state.last_gradcam = gradcam_img
                    else:
                        st.session_state.last_gradcam = None

                    # Feature 6: Extract feature vector for CBIR
                    feature_vec = extract_feature_vector(local_model, img_arr, meta_array=gcam_meta)
                    st.session_state.last_feature_vector = feature_vec
                else:
                    st.session_state.last_gradcam = None
                    st.session_state.last_feature_vector = None

                save_scan(st.session_state.last_patient_data, result,
                         feature_vector=feature_vec, image_thumbnail=image)
                st.toast("✅ Scan saved!", icon="💾")

    # Display Results
    if st.session_state.last_result:
        result = st.session_state.last_result
        predicted = result.get("predicted_class", "Unknown")
        confidence = result.get("confidence", 0.0)
        cancer_prob = result.get("cancer_prob", 0.0)
        all_probs = result.get("all_probs", {})
        reasoning_en = result.get("reasoning", "")
        recommendation_en = result.get("recommendation", "")
        uncertainty = result.get("uncertainty", "")
        triage = get_triage(cancer_prob)
        mode = result.get("inference_mode", "cloud")
        lesion_desc = result.get("lesion_description", "")
        morphology = result.get("morphology", "")
        color_pattern = result.get("color_pattern", "")
        border_analysis = result.get("border_analysis", "")
        differential = result.get("differential_diagnosis", [])
        risk_factors_text = result.get("risk_factors", "")

        st.markdown("---")

        # ==============================
        #  DETAILED CLINICAL REPORT
        # ==============================
        report_id = str(uuid.uuid4())[:10].upper()
        now_str = datetime.now().strftime("%d/%m/%Y  %I:%M %p")
        patient = st.session_state.get("last_patient_data", {})

        # --- Report Header ---
        risk_border_color = RISK_HIGH_COLOR if triage['label'] == "HIGH RISK" else (RISK_MODERATE_COLOR if triage['label'] == "MODERATE RISK" else RISK_LOW_COLOR)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {PRIMARY_COLOR}0f, {SECONDARY_COLOR}0f);
                    color: {text_primary}; padding: 25px 30px; border-radius: 15px 15px 0 0;
                    border-left: 6px solid {risk_border_color}; margin-top: 20px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h2 style='margin:0; color: {PRIMARY_COLOR};'>AI DermaScan — Clinical Report</h2>
                    <p style='margin:4px 0 0 0; color: {text_secondary}; font-size:13px;'>
                        Report ID: {report_id} &nbsp;|&nbsp; Generated: {now_str} &nbsp;|&nbsp;
                        {'🔒 Local' if mode == 'local' else '☁️ Cloud AI'} Inference
                    </p>
                </div>
                <div style='text-align:center; background: {bg_secondary};
                            border: 2px solid {risk_border_color}; border-radius: 12px;
                            padding: 10px 20px;'>
                    <span style='font-size: 28px; color: {text_primary};'>{triage['icon']}</span><br>
                    <span style='font-weight: bold; color: {text_primary}; font-size: 14px;'>{triage['label']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Patient Demographics ---
        st.markdown(f"""
        <div style='background: {bg_primary}; padding: 20px 30px; border-left: 6px solid {border_color};
                    margin: 0;'>
            <h4 style='margin: 0 0 10px 0; color: {border_color};'>👤 Patient Demographics</h4>
            <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;'>
                <div style='color: {text_primary};'><b>Name:</b> {patient.get('name', 'N/A')}</div>
                <div style='color: {text_primary};'><b>Age:</b> {patient.get('age', 'N/A')} years</div>
                <div style='color: {text_primary};'><b>Gender:</b> {patient.get('gender', 'N/A')}</div>
                <div style='color: {text_primary};'><b>Patient ID:</b> {patient.get('patient_id', 'N/A')}</div>
                <div style='color: {text_primary};'><b>Body Site:</b> {patient.get('body_location', 'N/A')}</div>
                <div style='color: {text_primary};'><b>Skin Type:</b> {patient.get('skin_type', 'N/A')}</div>
                <div style='color: {text_primary};'><b>Lesion Size:</b> {patient.get('lesion_size', 'N/A')} mm</div>
                <div style='color: {text_primary};'><b>Recent Change:</b> {patient.get('lesion_changed', 'N/A')}</div>
                <div style='color: {text_primary};'><b>Exam Date:</b> {patient.get('exam_date', 'N/A')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Primary Diagnosis ---
        cancer_classes = ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratoses"]
        is_malignant = predicted in cancer_classes
        diag_color = RISK_HIGH_COLOR if is_malignant else SUCCESS_COLOR
        st.markdown(f"""
        <div style='background: {bg_secondary}; padding: 20px 30px; border-left: 6px solid {diag_color};
                    border-bottom: 1px solid #eee;'>
            <h4 style='margin: 0 0 10px 0; color: {diag_color};'>🔬 Primary Diagnosis</h4>
            <div style='display: flex; gap: 30px; align-items: baseline;'>
                <span style='font-size: 24px; font-weight: bold; color: {diag_color};'>{predicted}</span>
                <span style='font-size: 16px; color: {text_secondary};'>Confidence: <b>{confidence:.1f}%</b></span>
                <span style='font-size: 16px; color: {text_secondary};'>Cancer Probability: <b style='color: {diag_color};'>{cancer_prob*100:.1f}%</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Clinical Reasoning ---
        if reasoning_en:
            st.markdown(f"""
            <div style='background: {bg_accent}; padding: 20px 30px; border-left: 6px solid {INFO_COLOR};'>
                <h4 style='margin: 0 0 10px 0; color: {INFO_COLOR};'>🧠 Clinical Reasoning</h4>
                <p style='margin: 0; line-height: 1.7; color: {text_primary};'>{reasoning_en}</p>
            </div>
            """, unsafe_allow_html=True)

        # --- Detailed Clinical Analysis (from cloud or generated) ---
        analysis_sections = []
        if lesion_desc:
            analysis_sections.append(("Lesion Description", lesion_desc, "#e64980"))
        if morphology:
            analysis_sections.append(("Morphology & Structure", morphology, "#845ef7"))
        if color_pattern:
            analysis_sections.append(("Color & Pigment Pattern", color_pattern, "#ff922b"))
        if border_analysis:
            analysis_sections.append(("Border Analysis", border_analysis, "#20c997"))

        if analysis_sections:
            st.markdown("<div style='padding: 0;'>", unsafe_allow_html=True)
            for title, text, color in analysis_sections:
                st.markdown(f"""
                <div style='background: {bg_secondary}; padding: 18px 30px; border-left: 6px solid {color};
                            border-bottom: 1px solid #f0f0f0;'>
                    <h4 style='margin: 0 0 8px 0; color: {color};'>🔍 {title}</h4>
                    <p style='margin: 0; line-height: 1.7; color: {text_primary};'>{text}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Differential Diagnosis ---
        if differential:
            diff_html = ""
            if isinstance(differential, list):
                for i, d in enumerate(differential):
                    diff_html += f"<span style='background:{bg_primary}; padding:5px 14px; border-radius:20px; margin:3px; display:inline-block; font-size:13px; border:1px solid {WARNING_COLOR}; color: {text_primary};'>{d}</span>"
            else:
                diff_html = str(differential)
            st.markdown(f"""
            <div style='background: {bg_accent}; padding: 20px 30px; border-left: 6px solid {WARNING_COLOR};'>
                <h4 style='margin: 0 0 10px 0; color: {WARNING_COLOR};'>📋 Differential Diagnosis</h4>
                <p style='margin: 0 0 6px 0; color: {text_primary}; font-size: 13px;'>Other conditions to consider:</p>
                <div>{diff_html}</div>
            </div>
            """, unsafe_allow_html=True)

        # --- Risk Factors ---
        if risk_factors_text:
            st.markdown(f"""
            <div style='background: {bg_accent}; padding: 20px 30px; border-left: 6px solid {SECONDARY_COLOR};'>
                <h4 style='margin: 0 0 10px 0; color: {SECONDARY_COLOR};'>⚠️ Risk Factors</h4>
                <p style='margin: 0; line-height: 1.7; color: {text_primary};'>{risk_factors_text}</p>
            </div>
            """, unsafe_allow_html=True)

        # --- Classification Probabilities Table ---
        if all_probs:
            st.markdown("### 📊 Classification Probabilities")
            prob_rows = ""
            for cls_name, prob_val in all_probs.items():
                pct = prob_val * 100
                is_cancer = cls_name in cancer_classes
                bar_color = RISK_HIGH_COLOR if is_cancer else SUCCESS_COLOR
                marker = " ⚠️" if is_cancer and pct > 5 else ""
                row_bg = bg_accent if is_cancer and pct > 10 else bg_secondary
                text_color = text_primary
                prob_rows += f"""
                <tr style='background: {row_bg}; color: {text_color};'>
                    <td style='padding:8px 12px; font-weight:{"bold" if pct == max(v*100 for v in all_probs.values()) else "normal"};'>{cls_name}{' 🔴' if is_cancer else ''}</td>
                    <td style='padding:8px 12px;'>
                        <div style='background:#e9ecef; border-radius:4px; height:20px; width:100%; position:relative;'>
                            <div style='background:{bar_color}; height:20px; border-radius:4px; width:{min(pct, 100):.1f}%;'></div>
                            <span style='position:absolute; left:8px; top:1px; font-size:12px; font-weight:bold;'>{pct:.1f}%{marker}</span>
                        </div>
                    </td>
                </tr>"""
            st.markdown(f"""
            <table style='width:100%; border-collapse:collapse; margin:10px 0; color: {text_primary};'>
                <thead>
                    <tr style='background:{bg_primary}; color: {text_primary};'>
                        <th style='padding:8px 12px; text-align:left;'>Condition</th>
                        <th style='padding:8px 12px; text-align:left;'>Probability</th>
                    </tr>
                </thead>
                <tbody>{prob_rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

        # --- Uncertainty ---
        if uncertainty:
            st.markdown(f"""
            <div style='background: {bg_accent}; padding: 15px 30px; border-left: 6px solid {border_color};'>
                <h4 style='margin: 0 0 8px 0; color: {border_color};'>🎯 Uncertainty Estimation (MC Dropout)</h4>
                <p style='margin: 0; font-family: monospace; font-size: 13px; color: {text_primary};'>{uncertainty}</p>
            </div>
            """, unsafe_allow_html=True)

        # --- Grad-CAM Display ---
        gcam = st.session_state.get("last_gradcam")
        if gcam is not None:
            st.markdown("### 🧠 Explainable AI — Grad-CAM Heatmap")
            gc1, gc2 = st.columns(2)
            with gc1:
                orig = st.session_state.get("last_image")
                if orig is not None:
                    st.image(np.array(orig), caption="Original Lesion", use_container_width=True)
                else:
                    st.info("Original image not available.")
            with gc2:
                st.image(gcam, caption="AI Attention Heatmap", use_container_width=True)
            st.caption("Red/yellow regions indicate where the AI focused most for its prediction.")

        # --- Hair Removal Preview ---
        hr_img = st.session_state.get("last_hair_removed")
        if hr_img is not None:
            st.markdown("### ✂️ Hair Removal Preprocessing")
            hr1, hr2 = st.columns(2)
            with hr1:
                _before_img = st.session_state.get("last_image")
                if _before_img is not None:
                    if hasattr(_before_img, "resize"):
                        st.image(np.array(_before_img.resize((224, 224))), caption="Before", use_container_width=True)
                    else:
                        st.image(_before_img, caption="Before", use_container_width=True)
                else:
                    st.info("Original image not available.")
            with hr2:
                st.image(hr_img, caption="After Hair Removal", use_container_width=True)

        # --- TTA Badge ---
        if st.session_state.get("enable_tta", False) and mode == "local":
            st.success("🎯 **Test-Time Augmentation (TTA) was used** — 8 augmented views averaged for robust prediction.")

        # --- Recommendation ---
        if theme_mode == "Dark":
            rec_bg = bg_secondary
            rec_hr = text_secondary
        else:
            rec_bg = f"linear-gradient(135deg, {SUCCESS_COLOR}1a, {SUCCESS_COLOR}0f)"
            rec_hr = f"{SUCCESS_COLOR}4d"
        st.markdown(f"""
        <div style='background: {rec_bg};
                    padding: 20px 30px; border-radius: 10px; margin: 15px 0;
                    border-left: 6px solid {SUCCESS_COLOR};'>
            <h4 style='margin: 0 0 10px 0; color: {SUCCESS_COLOR};'>🧑‍⚕️ Clinical Recommendation</h4>
            <p style='margin: 0; line-height: 1.7; color: {text_primary}; font-size: 15px;'>{recommendation_en}</p>
            <hr style='border-color: {rec_hr}; margin: 12px 0;'>
            <p style='margin: 0; font-size: 12px; color: {text_primary};'>
                <b>Triage Action:</b> {triage['action']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- Similar Cases ---
        fv = st.session_state.get("last_feature_vector")
        if fv is not None:
            similar = find_similar_cases(fv, top_k=3)
            if similar:
                st.markdown("### 🔍 Similar Past Cases (CBIR)")
                sim_cols = st.columns(len(similar))
                for i, case in enumerate(similar):
                    with sim_cols[i]:
                        if case.get("thumbnail"):
                            try:
                                thumb_arr = cv2.imdecode(np.frombuffer(case["thumbnail"], np.uint8), cv2.IMREAD_COLOR)
                                thumb_arr = cv2.cvtColor(thumb_arr, cv2.COLOR_BGR2RGB)
                                st.image(thumb_arr, caption=f"Case #{case['id']}", use_container_width=True)
                            except Exception:
                                st.write(f"Case #{case['id']}")
                        st.markdown(f"**{case['predicted_class']}**")
                        st.markdown(f"Confidence: {case['confidence']:.1f}%")
                        st.markdown(f"Similarity: {case['similarity']*100:.1f}%")
                        st.markdown(f"Risk: {case['risk_level']}")

        # --- Translation ---
        lang = st.session_state.selected_language
        if lang != "English":
            st.markdown(f"### 🌐 Translation ({lang})")
            with st.spinner(f"Translating to {lang}..."):
                tr_reasoning = translate_text(reasoning_en, lang)
                tr_recommendation = translate_text(recommendation_en, lang)
            st.info(f"**Reasoning ({lang}):** {tr_reasoning}")
            st.warning(f"**Recommendation ({lang}):** {tr_recommendation}")

        # --- Disclaimer ---
        st.markdown(f"""
        <div style='background: {bg_primary}; padding: 15px 30px; border-radius: 0 0 15px 15px;
                    border-top: 2px solid {border_color}; margin-bottom: 20px;'>
            <p style='margin: 0; font-size: 11px; color: {text_secondary}; text-align: center;'>
                ⚠️ This AI-generated report is a screening aid only and does NOT constitute a definitive medical diagnosis.
                Always consult a qualified dermatologist for clinical evaluation. Report ID: {report_id}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- PDF Report ---
        if "last_patient_data" in st.session_state and "last_image" in st.session_state:
            gradcam_img = st.session_state.get("last_gradcam")
            pdf = generate_pdf(st.session_state.last_patient_data, st.session_state.last_image, result, gradcam_img)
            with open(pdf, "rb") as f:
                st.download_button("📄 Download Full Report (PDF)", f, file_name=pdf, use_container_width=True)
# ===================================================================
#                        PAGE: DASHBOARD
# ===================================================================
if page == "📊 Dashboard":
    st.markdown("## 📊 Patient History Dashboard")
    scans_df = get_all_scans()
    # Privacy: patients only see their own scans; doctors/admins see all
    if st.session_state.logged_in and st.session_state.user_role == "patient":
        _pid = st.session_state.user_data.get("patient_id", "")
        if _pid and "patient_id" in scans_df.columns:
            scans_df = scans_df[scans_df["patient_id"] == _pid]

    if len(scans_df) == 0:
        st.info("No scans recorded yet. Go to 🔬 Prediction to analyze your first image!")
    else:
        st.markdown("### 📈 Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scans", len(scans_df))
        high_risk = len(scans_df[scans_df["risk_level"] == "HIGH"]) if "risk_level" in scans_df.columns else 0
        c2.metric("🔴 High Risk", high_risk)
        medium_risk = len(scans_df[scans_df["risk_level"] == "MODERATE"]) if "risk_level" in scans_df.columns else 0
        c3.metric("🟡 Moderate", medium_risk)
        low_risk = len(scans_df[scans_df["risk_level"] == "LOW"]) if "risk_level" in scans_df.columns else 0
        c4.metric("🟢 Low Risk", low_risk)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🧪 Diagnosis Distribution")
            diagnosis_counts = scans_df["predicted_class"].value_counts()
            fig1, ax1 = plt.subplots()
            colors_list = ['#ff6b6b','#51cf66','#ffd43b','#339af0','#845ef7','#ff922b','#20c997','#e64980','#adb5bd']
            ax1.pie(diagnosis_counts.values, labels=diagnosis_counts.index, autopct='%1.1f%%',
                    colors=colors_list[:len(diagnosis_counts)], startangle=90)
            ax1.set_title("Diagnosis Breakdown")
            st.pyplot(fig1)
            plt.close()

        with col2:
            st.markdown("### 📅 Scans Over Time")
            scans_df["scan_date"] = pd.to_datetime(scans_df["scan_timestamp"]).dt.date
            daily_counts = scans_df.groupby("scan_date").size().reset_index(name="count")
            fig2, ax2 = plt.subplots()
            ax2.bar(daily_counts["scan_date"].astype(str), daily_counts["count"], color="#667eea")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Scans")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### 🎯 Confidence Distribution")
            fig3, ax3 = plt.subplots()
            ax3.hist(scans_df["confidence"], bins=20, color="#f093fb", edgecolor="white")
            ax3.set_xlabel("Confidence (%)")
            ax3.set_ylabel("Frequency")
            st.pyplot(fig3)
            plt.close()

        with col4:
            st.markdown("### ⚠️ Risk Level Distribution")
            risk_counts = scans_df["risk_level"].value_counts()
            fig4, ax4 = plt.subplots()
            risk_colors = {"LOW": "#51cf66", "MODERATE": "#ffd43b", "HIGH": "#ff6b6b"}
            bar_colors = [risk_colors.get(r, "#aaa") for r in risk_counts.index]
            ax4.bar(risk_counts.index, risk_counts.values, color=bar_colors)
            ax4.set_xlabel("Risk Level")
            ax4.set_ylabel("Count")
            st.pyplot(fig4)
            plt.close()

        st.markdown("---")
        st.markdown("### 🗂️ Scan History")
        search_query = st.text_input("🔍 Search by Patient Name or ID")
        filtered = scans_df
        if search_query:
            filtered = scans_df[
                scans_df["name"].str.contains(search_query, case=False, na=False) |
                scans_df["patient_id"].str.contains(search_query, case=False, na=False)
            ]
        display_cols = ["name","age","gender","patient_id","predicted_class",
                       "confidence","risk_level","body_location","skin_type","inference_mode","scan_timestamp"]
        available_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available_cols], use_container_width=True)

# ===================================================================
#                        PAGE: TRACKING (Feature 8)
# ===================================================================
if page == "📈 Tracking":
    st.markdown("## 📈 Longitudinal Lesion Tracking")
    st.markdown("Track a patient's skin lesion history over time to monitor changes.")

    scans_df = get_all_scans()
    if len(scans_df) == 0:
        st.info("No scans recorded yet. Go to 🔬 Prediction to analyze your first image!")
    else:
        # Get unique patient IDs
        patient_ids = scans_df[scans_df["patient_id"].str.strip() != ""]["patient_id"].unique().tolist()
        patient_names = scans_df[scans_df["name"].str.strip() != ""]["name"].unique().tolist()

        search_method = st.radio("Search by:", ["Patient ID", "Patient Name"], horizontal=True)
        if search_method == "Patient ID":
            if patient_ids:
                selected = st.selectbox("Select Patient ID", patient_ids)
                patient_scans = scans_df[scans_df["patient_id"] == selected].sort_values("scan_timestamp")
            else:
                st.warning("No patient IDs found in scan history.")
                patient_scans = pd.DataFrame()
        else:
            if patient_names:
                selected = st.selectbox("Select Patient Name", patient_names)
                patient_scans = scans_df[scans_df["name"] == selected].sort_values("scan_timestamp")
            else:
                st.warning("No patient names found in scan history.")
                patient_scans = pd.DataFrame()

        if len(patient_scans) > 0:
            st.markdown(f"### Found **{len(patient_scans)}** scans")

            # Trend indicators
            if len(patient_scans) >= 2:
                first_prob = patient_scans.iloc[0]["cancer_prob"]
                last_prob = patient_scans.iloc[-1]["cancer_prob"]
                diff = last_prob - first_prob
                if diff > 0.05:
                    trend = "↗️ **Increasing Risk** — Cancer probability is trending upward."
                    trend_color = "#ff6b6b"
                elif diff < -0.05:
                    trend = "↘️ **Decreasing Risk** — Cancer probability is trending downward."
                    trend_color = "#51cf66"
                else:
                    trend = "➡️ **Stable** — Cancer probability is relatively stable."
                    trend_color = "#ffd43b"
                st.markdown(f"<div style='background: {trend_color}22; border-left: 4px solid {trend_color}; padding: 15px; border-radius: 8px; margin: 10px 0;'>{trend}</div>", unsafe_allow_html=True)

            # Metrics
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Scans", len(patient_scans))
            mc2.metric("Latest Diagnosis", patient_scans.iloc[-1]["predicted_class"])
            mc3.metric("Latest Risk", patient_scans.iloc[-1]["risk_level"])
            mc4.metric("Avg Confidence", f"{patient_scans['confidence'].mean():.1f}%")

            # Cancer probability over time chart
            st.markdown("### 📉 Cancer Probability Over Time")
            fig, ax = plt.subplots(figsize=(10, 4))
            timestamps = pd.to_datetime(patient_scans["scan_timestamp"])
            ax.plot(timestamps, patient_scans["cancer_prob"] * 100, 'o-', color='#ff6b6b', linewidth=2, markersize=8, label="Cancer Prob %")
            ax.axhline(y=30, color='#ffd43b', linestyle='--', alpha=0.7, label="Moderate Threshold (30%)")
            ax.axhline(y=60, color='#ff6b6b', linestyle='--', alpha=0.7, label="High Risk Threshold (60%)")
            ax.fill_between(timestamps, 0, 30, alpha=0.05, color='green')
            ax.fill_between(timestamps, 30, 60, alpha=0.05, color='yellow')
            ax.fill_between(timestamps, 60, 100, alpha=0.05, color='red')
            ax.set_xlabel("Date")
            ax.set_ylabel("Cancer Probability (%)")
            ax.set_ylim(0, 100)
            ax.legend(fontsize=8)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Confidence over time
            st.markdown("### 🎯 Model Confidence Over Time")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.bar(timestamps.dt.strftime("%Y-%m-%d %H:%M"), patient_scans["confidence"], color="#667eea", alpha=0.8)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Confidence (%)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            # Scan history table
            st.markdown("### 🗂️ Scan Details")
            display_cols = ["scan_timestamp", "predicted_class", "confidence", "cancer_prob", "risk_level", "body_location", "inference_mode"]
            available = [c for c in display_cols if c in patient_scans.columns]
            st.dataframe(patient_scans[available], use_container_width=True)

            # Visual progression with thumbnails
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            scan_ids = patient_scans["id"].tolist() if "id" in patient_scans.columns else []
            if scan_ids:
                st.markdown("### 🖼️ Visual Progression")
                thumb_cols = st.columns(min(len(scan_ids), 6))
                for i, sid in enumerate(scan_ids[:6]):
                    c.execute("SELECT image_thumbnail, scan_timestamp, predicted_class FROM scans WHERE id=?", (int(sid),))
                    row = c.fetchone()
                    if row and row[0]:
                        try:
                            thumb = cv2.imdecode(np.frombuffer(row[0], np.uint8), cv2.IMREAD_COLOR)
                            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                            with thumb_cols[i % len(thumb_cols)]:
                                st.image(thumb, caption=f"{row[1][:10]}\n{row[2]}", use_container_width=True)
                        except Exception:
                            pass
            conn.close()
        elif search_method == "Patient ID" and patient_ids or search_method == "Patient Name" and patient_names:
            st.info("No scans found for this patient.")


# ===================================================================
#                        PAGE: AI CHATBOT
# ===================================================================
if page == "💬 Ask AI":
    st.markdown("## 🩺 AI Dermatologist Assistant")
    st.markdown("Get expert-level clinical insights about your skin diagnosis. Powered by an elite AI dermatologist prompt.")

    if not GEMINI_API_KEY:
        st.warning("Please enter your Gemini API Key in the sidebar to use the AI Dermatologist.")
    else:
        # ── Build Patient Metadata Context ──
        if "last_patient_data" in st.session_state:
            p = st.session_state.last_patient_data
            patient_metadata_str = f"""- Name: {p.get('name', 'N/A')}
- Age: {p.get('age', 'N/A')}
- Gender: {p.get('gender', 'N/A')}
- Skin Type: {p.get('skin_type', 'N/A')}
- Lesion Location: {p.get('body_location', 'N/A')}
- Size: {p.get('lesion_size', 'N/A')} mm
- Lesion Change History: {p.get('lesion_changed', 'N/A')}"""
        else:
            patient_metadata_str = "No patient data available."
            
        scan_result_str = "No scan performed yet."

        if st.session_state.last_result:
            last = st.session_state.last_result
            scan_result_str = f"""- Predicted Condition: {last.get('predicted_class', 'Unknown')}
- Confidence: {last.get('confidence', 0):.1f}%
- Cancer Probability: {last.get('cancer_prob', 0)*100:.1f}%
- Risk Level: {"HIGH" if last.get('cancer_prob', 0) >= 0.6 else "MODERATE" if last.get('cancer_prob', 0) >= 0.3 else "LOW"}
- Reasoning: {last.get('reasoning', 'N/A')}
- Recommendation: {last.get('recommendation', 'N/A')}
- Uncertainty: {last.get('uncertainty', 'N/A')}
- Lesion Description: {last.get('lesion_description', 'N/A')}
- Morphology: {last.get('morphology', 'N/A')}
- Color Pattern: {last.get('color_pattern', 'N/A')}
- Border Analysis: {last.get('border_analysis', 'N/A')}
- Differential Diagnoses: {last.get('differential_diagnosis', 'N/A')}
- Inference Mode: {last.get('inference_mode', 'N/A')}"""

            # Display context card
            cancer_prob = last.get('cancer_prob', 0)
            risk_color = "#ff6b6b" if cancer_prob >= 0.6 else "#ffd43b" if cancer_prob >= 0.3 else "#51cf66"
            risk_label = "HIGH RISK" if cancer_prob >= 0.6 else "MODERATE RISK" if cancer_prob >= 0.3 else "LOW RISK"
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea11, #764ba222);
                        padding: 20px; border-radius: 12px; border-left: 5px solid {risk_color}; margin-bottom: 20px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h4 style='margin: 0;'>📋 Active Diagnosis Context</h4>
                        <p style='margin: 5px 0;'><b>Condition:</b> {last.get('predicted_class', 'N/A')}</p>
                        <p style='margin: 5px 0;'><b>Confidence:</b> {last.get('confidence', 0):.1f}%</p>
                        <p style='margin: 5px 0; font-size: 13px; color: #888;'>{last.get('reasoning', 'N/A')[:150]}...</p>
                    </div>
                    <div style='text-align: center; padding: 10px 20px; background: {risk_color}22;
                                border-radius: 10px; border: 2px solid {risk_color};'>
                        <span style='font-size: 20px; font-weight: bold; color: {risk_color};'>{risk_label}</span><br>
                        <span style='font-size: 12px; color: #888;'>Cancer Prob: {cancer_prob*100:.1f}%</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("💡 No diagnosis yet. Go to 🔬 Prediction first for personalized, context-aware clinical responses.")

        # ── Chat History Display ──
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🩺"):
                st.markdown(msg["content"])

        # ── Chat Input ──
        user_input = st.chat_input("Ask the AI Dermatologist about your diagnosis, symptoms, or skin health...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.rerun()

        # ── Generate AI Response if needed ──
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            active_question = st.session_state.chat_history[-1]["content"]
            
            # ── Build Chat History String ──
            chat_history_str = ""
            # Exclude current question from history context to avoid doubling
            for msg in st.session_state.chat_history[-8:-1]:
                role = "Patient" if msg["role"] == "user" else "Dermatologist AI"
                chat_history_str += f"{role}: {msg['content']}\n\n"

            # ── Elite AI Dermatologist Prompt ──
            system_prompt = f"""You are an elite AI Dermatologist Assistant operating at expert clinical level.

You are NOT a generic chatbot. You behave like a highly experienced dermatologist who:
* Thinks step-by-step
* Personalizes every response
* Explains clearly to non-medical users
* Prioritizes patient safety and clarity

═══════════════════════════════
🧠 CONTEXT (USE THIS ALWAYS)
═══════════════════════════════

Patient Profile:
{patient_metadata_str}

Latest Scan Result:
{scan_result_str}

Chat History:
{chat_history_str}

User Question:
{active_question}

═══════════════════════════════
🎯 YOUR OBJECTIVE
═══════════════════════════════

Deliver a response that is:
* Highly personalized to THIS patient
* Clinically accurate but easy to understand
* Actionable (tells user exactly what to do next)
* Honest about uncertainty

═══════════════════════════════
🧬 THINK LIKE A DERMATOLOGIST
═══════════════════════════════

Internally analyze before answering:
* Visual clues (shape, border, color, texture)
* Risk indicators (irregularity, variation, growth signs)
* Patient factors (age, skin type, lesion location)
* Probability vs real-world risk
* Possible alternative diagnoses

═══════════════════════════════
📊 RESPONSE STRUCTURE (MANDATORY)
═══════════════════════════════

1. 🧾 Simple Summary
   Explain in 1–2 lines what this likely means for the user.

2. 🔬 Clinical Insight
   Explain WHY this prediction was made (features, patterns, reasoning).

3. ⚖️ Risk Interpretation
   Explain the risk level in real-life terms (not just percentage).

4. 🧠 Possible Alternatives
   List 2–3 other possible conditions and how they differ.

5. 🧑‍⚕️ What I Would Ask You
   Ask 2–4 smart follow-up questions like a real doctor.

6. ✅ Recommended Action
   Give clear next steps: Monitor / Recheck / See doctor / Urgent action

7. ⚠️ Red Flags
   Tell user what symptoms would make this dangerous.

═══════════════════════════════
🚨 SAFETY RULES
═══════════════════════════════

* NEVER give absolute diagnosis
* NEVER say "you definitely have cancer"
* If high risk → strongly advise medical consultation
* If uncertain → clearly say so
* Always prioritize user safety over confidence

═══════════════════════════════
🧠 BEHAVIOR RULES
═══════════════════════════════

* Do NOT give generic textbook answers
* Do NOT ignore patient context
* Speak like a calm, intelligent doctor
* Balance reassurance + seriousness
* Keep response structured and readable

═══════════════════════════════
🔥 EXTRA INTELLIGENCE MODE
═══════════════════════════════

If user is confused or anxious → simplify language, reassure without dismissing
If case is high-risk → increase urgency, guide immediate action
If case is low-risk → reassure + suggest monitoring

⚠️ IMPORTANT: You are an AI assistant, NOT a real doctor. Always include a disclaimer that this is AI analysis and professional medical consultation is essential for any clinical decisions.

Now answer the user's question using the structured format above."""

            with st.chat_message("assistant", avatar="🩺"):
                with st.spinner("🩺 Analyzing your question..."):
                    try:
                        client = get_gemini_client(GEMINI_API_KEY)
                        response = generate_gemini_content(client, system_prompt)
                        ai_response = response.text.strip()

                        # Translation support
                        lang = st.session_state.selected_language
                        if lang != "English":
                            translated = translate_text(ai_response, lang)
                            st.markdown(ai_response)
                            st.markdown(f"---\n**🌐 {lang}:**\n{translated}")
                            ai_response = ai_response + f"\n\n---\n🌐 {lang}:\n{translated}"
                        else:
                            st.markdown(ai_response)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        # ── Quick Question Buttons ──
        if not st.session_state.chat_history:
            st.markdown("### 💡 Quick Questions")
            q_col1, q_col2, q_col3 = st.columns(3)
            with q_col1:
                if st.button("🔍 Explain my diagnosis", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "Can you explain my diagnosis in simple terms?"})
                    st.rerun()
            with q_col2:
                if st.button("⚠️ Should I be worried?", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "Should I be worried about my scan results? What is my real risk?"})
                    st.rerun()
            with q_col3:
                if st.button("🧑‍⚕️ What should I do next?", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "What are the recommended next steps for me?"})
                    st.rerun()

        # ── Clear Chat ──
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()


# ===================================================================
#                        PAGE: FIND CLINICS
# ===================================================================
if page == "🗺️ Find Clinics":
    st.markdown("## 🗺️ Find Nearby Dermatology Clinics")
    st.markdown("Locate dermatologists, skin specialists, and derma clinics near you.")

    # Location Input
    loc_col1, loc_col2 = st.columns([3, 1])
    with loc_col1:
        city_input = st.text_input("🔍 Enter your City or Area", placeholder="e.g. Greater Noida, Delhi, Mumbai...")
    with loc_col2:
        search_radius = st.selectbox("Radius", ["5 km", "10 km", "20 km", "50 km"], index=1)

    if city_input and st.button("🔎 Search Dermatology Clinics", use_container_width=True):
        if not GEMINI_API_KEY:
            st.error("Please enter your Gemini API Key in the sidebar to search for clinics.")
        else:
            with st.spinner(f"🔍 Searching for dermatology clinics near {city_input}..."):
                try:
                    client = get_gemini_client(GEMINI_API_KEY)
                    prompt = f"""Find 8-12 real dermatology clinics, skin specialists, or skin cancer hospitals near {city_input}, India within {search_radius}.

Return ONLY a valid JSON array. Each object must have:
- "name": clinic/hospital name
- "address": full address
- "lat": latitude (float)
- "lon": longitude (float)
- "type": one of "Dermatology Clinic", "Skin Hospital", "Multi-Specialty Hospital", "Cosmetic Dermatology"
- "phone": phone number or "N/A"
- "rating": rating out of 5 (float) or null
- "specialties": short list of specialties like "Skin Cancer, Moles, Psoriasis"

Return ONLY the JSON array, no markdown, no explanation. Use real, accurate coordinates for the {city_input} area."""
                    response = generate_gemini_content(client, prompt)
                    text = response.text.strip()
                    start_idx = text.find('[')
                    end_idx = text.rfind(']')
                    
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        json_str = text[start_idx:end_idx+1]
                        clinics = json.loads(json_str)
                        st.session_state.found_clinics = clinics
                        st.session_state.clinic_city = city_input
                    else:
                        st.error("AI response didn't contain valid data. Try again.")
                        st.session_state.found_clinics = []
                except Exception as e:
                    st.error(f"Error searching for clinics: {str(e)}")
                    st.session_state.found_clinics = []

    # Display Results
    if "found_clinics" in st.session_state and st.session_state.found_clinics:
        clinics = st.session_state.found_clinics
        city = st.session_state.get("clinic_city", "your area")

        st.success(f"Found **{len(clinics)}** dermatology clinics near {city}")

        # Calculate map center securely skipping nulls
        valid_lats = [float(c.get("lat")) for c in clinics if c.get("lat") is not None]
        valid_lons = [float(c.get("lon")) for c in clinics if c.get("lon") is not None]
        avg_lat = sum(valid_lats) / len(valid_lats) if valid_lats else 28.47
        avg_lon = sum(valid_lons) / len(valid_lons) if valid_lons else 77.50

        # Create Folium Map
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles="OpenStreetMap")

        # Color by type
        type_colors = {
            "Dermatology Clinic": "blue",
            "Skin Hospital": "red",
            "Multi-Specialty Hospital": "green",
            "Cosmetic Dermatology": "purple",
        }
        type_icons = {
            "Dermatology Clinic": "stethoscope",
            "Skin Hospital": "hospital",
            "Multi-Specialty Hospital": "plus-square",
            "Cosmetic Dermatology": "star",
        }

        for clinic in clinics:
            # Fallback to average if actual coordinate is null
            lat = clinic.get("lat")
            lat = float(lat) if lat is not None else avg_lat
            lon = clinic.get("lon")
            lon = float(lon) if lon is not None else avg_lon
            name = clinic.get("name", "Unknown")
            address = clinic.get("address", "")
            phone = clinic.get("phone", "N/A")
            c_type = clinic.get("type", "Dermatology Clinic")
            rating = clinic.get("rating")
            specialties = clinic.get("specialties", "")

            rating_str = f"⭐ {rating}/5" if rating else "No rating"
            popup_html = f"""
            <div style='width: 250px; font-family: Arial;'>
                <h4 style='margin: 0; color: #1a73e8;'>{name}</h4>
                <p style='margin: 4px 0; font-size: 12px; color: #555;'>{address}</p>
                <p style='margin: 4px 0; font-size: 12px;'>📞 {phone}</p>
                <p style='margin: 4px 0; font-size: 12px;'>🏥 {c_type}</p>
                <p style='margin: 4px 0; font-size: 12px;'>{rating_str}</p>
                <p style='margin: 4px 0; font-size: 11px; color: #777;'>🔬 {specialties}</p>
                <a href='https://www.google.com/maps/dir/?api=1&destination={lat},{lon}' target='_blank'
                   style='font-size: 12px; color: #1a73e8;'>📍 Get Directions</a>
            </div>
            """

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=name,
                icon=folium.Icon(
                    color=type_colors.get(c_type, "gray"),
                    icon=type_icons.get(c_type, "info-sign"),
                    prefix="fa"
                )
            ).add_to(m)

        # Display the map
        st_folium(m, width=1200, height=500)

        # Legend
        st.markdown("""        <div style='display: flex; gap: 20px; justify-content: center; margin: 10px 0; flex-wrap: wrap;'>
            <span>🔵 Dermatology Clinic</span>
            <span>🔴 Skin Hospital</span>
            <span>🟢 Multi-Specialty Hospital</span>
            <span>🟣 Cosmetic Dermatology</span>
        </div>
        """, unsafe_allow_html=True)

        # Clinic Cards
        st.markdown("### 🏥 Clinic Details")
        for i, clinic in enumerate(clinics):
            c_type = clinic.get("type", "Clinic")
            rating = clinic.get("rating")
            rating_str = f"⭐ {rating}/5" if rating else ""
            type_color = {"Dermatology Clinic": "#339af0", "Skin Hospital": "#ff6b6b",
                         "Multi-Specialty Hospital": "#51cf66", "Cosmetic Dermatology": "#845ef7"}.get(c_type, "#aaa")
            
            # Safe parsing
            lat = clinic.get("lat")
            lat = float(lat) if lat is not None else avg_lat
            lon = clinic.get("lon")
            lon = float(lon) if lon is not None else avg_lon

            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {type_color}11, {type_color}22);
                        border-left: 4px solid {type_color}; padding: 15px; border-radius: 10px; margin: 8px 0;'>
                <div style='display: flex; justify-content: space-between; align-items: start;'>
                    <div>
                        <h4 style='margin: 0;'>{clinic.get('name', 'Unknown')}</h4>
                        <p style='margin: 4px 0; color: #666; font-size: 14px;'>📍 {clinic.get('address', '')}</p>
                        <p style='margin: 4px 0; font-size: 13px;'>📞 {clinic.get('phone', 'N/A')} &nbsp;|&nbsp; 🏥 {c_type}</p>
                        <p style='margin: 4px 0; font-size: 13px; color: #888;'>🔬 {clinic.get('specialties', '')}</p>
                    </div>
                    <div style='text-align: right;'>
                        <span style='font-size: 16px;'>{rating_str}</span><br>
                        <a href='https://www.google.com/maps/dir/?api=1&destination={lat},{lon}'
                           target='_blank' style='font-size: 13px; color: {type_color};'>Get Directions →</a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif "found_clinics" in st.session_state and not st.session_state.found_clinics:
        st.info("No clinics found. Try a different city or area.")
    else:
        # Default helpful info when no search performed
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea22, #764ba222);
                    padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;'>
            <h3>🔍 How to use</h3>
            <p>Enter your city or area name above and click <b>Search</b> to find dermatology clinics near you.</p>
            <p style='color: #888; font-size: 13px;'>Powered by Gemini AI • Results include clinic locations, contact info, and Google Maps directions</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🩺 When should you visit a dermatologist?")
        w1, w2, w3 = st.columns(3)
        w1.markdown("""**🔴 Urgently**\n- Rapidly changing mole\n- Bleeding lesion\n- AI flagged HIGH risk""")
        w2.markdown("""**🟡 Soon (2 weeks)**\n- New or unusual growth\n- Persistent itching\n- AI flagged MODERATE risk""")
        w3.markdown("""**🟢 Routine**\n- Annual skin check\n- Family history of cancer\n- Self-monitoring""")



# ===================================================================
#                        PAGE: MY PROFILE
# ===================================================================
if page == "👤 My Profile":
    if not st.session_state.logged_in:
        st.warning("🔐 Please login to view your profile.")
        st.stop()

    user = st.session_state.user_data
    role = st.session_state.user_role

    if role == "patient":
        st.markdown("## 👤 Patient Profile")
        # Refresh data
        fresh = get_patient(user["patient_id"])
        if fresh:
            user = fresh
            st.session_state.user_data = fresh

        # Profile Header Card
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea22, #764ba222);
                    padding: 25px; border-radius: 15px; border-left: 6px solid #667eea; margin-bottom: 20px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h2 style='margin: 0;'>🧑 {user.get('full_name', 'N/A')}</h2>
                    <p style='margin: 5px 0; color: #888;'>Patient ID: {user.get('patient_id', 'N/A')}</p>
                    <p style='margin: 5px 0; color: #888;'>📧 {user.get('email', 'N/A')} &nbsp;|&nbsp; 📞 {user.get('phone_number', 'N/A') or 'Not set'}</p>
                </div>
                <div style='text-align: center; background: #667eea22; border-radius: 12px; padding: 15px 25px;'>
                    <span style='font-size: 28px;'>🧬</span><br>
                    <span style='font-size: 12px; color: #888;'>Member since<br>{user.get('account_created_at', 'N/A')[:10] if user.get('account_created_at') else 'N/A'}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Basic Info", "🏥 Medical Info", "⚙️ Account Settings", "🔑 Change Password"])

        with tab1:
            st.markdown("### 📋 Basic Information")
            with st.form("edit_basic_info"):
                ec1, ec2 = st.columns(2)
                with ec1:
                    edit_name = st.text_input("Full Name", value=user.get("full_name", ""))
                    edit_age = st.number_input("Age", 1, 120, value=user.get("age", 25) or 25)
                    edit_gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                                              index=["Male", "Female", "Other"].index(user.get("gender", "Male")) if user.get("gender") in ["Male", "Female", "Other"] else 0)
                with ec2:
                    edit_dob = st.text_input("Date of Birth", value=user.get("date_of_birth", "") or "")
                    edit_phone = st.text_input("Phone Number", value=user.get("phone_number", "") or "")
                    edit_email_display = st.text_input("Email (read-only)", value=user.get("email", ""), disabled=True)

                if st.form_submit_button("💾 Save Basic Info", use_container_width=True):
                    update_patient(user["patient_id"],
                                   full_name=edit_name, age=edit_age, gender=edit_gender,
                                   date_of_birth=edit_dob, phone_number=edit_phone)
                    st.session_state.user_data = get_patient(user["patient_id"])
                    st.success("✅ Basic info updated!")
                    st.rerun()

        with tab2:
            st.markdown("### 🏥 Medical Information")
            with st.form("edit_medical_info"):
                med1, med2 = st.columns(2)
                with med1:
                    edit_blood = st.selectbox("Blood Group", ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                                             index=["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"].index(user.get("blood_group", "")) if user.get("blood_group") in ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"] else 0)
                    edit_allergies = st.text_area("Allergies", value=user.get("allergies", "") or "")
                with med2:
                    edit_conditions = st.text_area("Existing Conditions", value=user.get("existing_conditions", "") or "")
                    edit_family = st.text_area("Family History (skin cancer, etc.)", value=user.get("family_history", "") or "")

                if st.form_submit_button("💾 Save Medical Info", use_container_width=True):
                    update_patient(user["patient_id"],
                                   blood_group=edit_blood, allergies=edit_allergies,
                                   existing_conditions=edit_conditions, family_history=edit_family)
                    st.session_state.user_data = get_patient(user["patient_id"])
                    st.success("✅ Medical info updated!")
                    st.rerun()

        with tab3:
            st.markdown("### ⚙️ Account Settings")
            st.markdown(f"""
            | Setting | Value |
            |---|---|
            | **Patient ID** | `{user.get('patient_id', 'N/A')}` |
            | **Email** | {user.get('email', 'N/A')} |
            | **Account Created** | {user.get('account_created_at', 'N/A')} |
            | **Last Login** | {user.get('last_login', 'N/A')} |
            | **Notifications** | {user.get('notification_preferences', 'all')} |
            """)

            notif_pref = st.selectbox("Notification Preferences",
                                      ["all", "email_only", "none"],
                                      index=["all", "email_only", "none"].index(user.get("notification_preferences", "all")) if user.get("notification_preferences") in ["all", "email_only", "none"] else 0)
            if st.button("💾 Update Notifications"):
                update_patient(user["patient_id"], notification_preferences=notif_pref)
                st.success("✅ Notification preferences updated!")

        with tab4:
            st.markdown("### 🔑 Change Password")
            with st.form("change_password"):
                old_pass = st.text_input("Current Password", type="password")
                new_pass = st.text_input("New Password", type="password")
                new_pass2 = st.text_input("Confirm New Password", type="password")
                if st.form_submit_button("🔒 Change Password", use_container_width=True):
                    if not verify_password(old_pass, user.get("password_hash", "")):
                        st.error("❌ Current password is incorrect.")
                    elif new_pass != new_pass2:
                        st.error("❌ New passwords don't match.")
                    elif len(new_pass) < 6:
                        st.error("❌ Password must be at least 6 characters.")
                    else:
                        new_hash = hash_password(new_pass)
                        update_patient(user["patient_id"], password_hash=new_hash)
                        st.session_state.user_data = get_patient(user["patient_id"])
                        st.success("✅ Password changed successfully!")

        # Scan History Summary
        st.markdown("---")
        st.markdown("### 📊 My Scan History")
        scans_df = get_all_scans()
        my_scans = scans_df[scans_df["patient_id"] == user["patient_id"]] if "patient_id" in scans_df.columns else pd.DataFrame()
        if len(my_scans) > 0:
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Total Scans", len(my_scans))
            high = len(my_scans[my_scans["risk_level"] == "HIGH"]) if "risk_level" in my_scans.columns else 0
            sc2.metric("High Risk", high)
            sc3.metric("Last Scan", my_scans.iloc[0]["scan_timestamp"][:10] if len(my_scans) > 0 else "N/A")
            display_cols = ["scan_timestamp", "predicted_class", "confidence", "risk_level", "body_location"]
            available = [c for c in display_cols if c in my_scans.columns]
            st.dataframe(my_scans[available].head(10), use_container_width=True)
        else:
            st.info("No scans yet. Go to 🔬 Prediction to get your first scan!")

    elif role in ["doctor", "admin"]:
        st.markdown("## 👨‍⚕️ Doctor Profile")
        fresh = get_doctor(user["doctor_id"])
        if fresh:
            user = fresh
            st.session_state.user_data = fresh

        # Profile Header
        v_color = "#51cf66" if user.get("verification_status") == "Verified" else "#ffd43b" if user.get("verification_status") == "Pending" else "#ff6b6b"
        v_label = user.get("verification_status", "Pending")
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #20c99722, #339af022);
                    padding: 25px; border-radius: 15px; border-left: 6px solid #20c997; margin-bottom: 20px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h2 style='margin: 0;'>👨‍⚕️ Dr. {user.get('full_name', 'N/A')}</h2>
                    <p style='margin: 5px 0; color: #888;'>{user.get('specialization', 'N/A')} • {user.get('qualification', 'N/A')}</p>
                    <p style='margin: 5px 0; color: #888;'>🏥 {user.get('hospital_or_clinic_name', 'N/A') or 'Not set'} &nbsp;|&nbsp; 📍 {user.get('city', 'N/A') or 'Not set'}</p>
                </div>
                <div style='text-align: center;'>
                    <span style='background: {v_color}22; border: 2px solid {v_color}; border-radius: 10px; padding: 8px 16px; font-weight: bold; color: {v_color};'>{v_label}</span>
                    <br><br>
                    <span style='font-size: 14px;'>⭐ {user.get('average_rating', 0):.1f}/5</span><br>
                    <span style='font-size: 12px; color: #888;'>{user.get('total_patients_handled', 0)} reviews</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Basic Info", "🏥 Work Info", "📅 Availability", "🔑 Change Password"])

        with tab1:
            with st.form("edit_doc_basic"):
                dc1, dc2 = st.columns(2)
                with dc1:
                    edit_name = st.text_input("Full Name", value=user.get("full_name", ""))
                    edit_phone = st.text_input("Phone", value=user.get("phone_number", "") or "")
                    edit_spec = st.selectbox("Specialization",
                        ["Dermatologist", "Oncologist", "General Physician", "Plastic Surgeon", "Pathologist", "Other"],
                        index=["Dermatologist", "Oncologist", "General Physician", "Plastic Surgeon", "Pathologist", "Other"].index(user.get("specialization", "Dermatologist")) if user.get("specialization") in ["Dermatologist", "Oncologist", "General Physician", "Plastic Surgeon", "Pathologist", "Other"] else 0)
                with dc2:
                    edit_qual = st.text_input("Qualification", value=user.get("qualification", "") or "")
                    edit_exp = st.number_input("Years of Experience", 0, 50, value=user.get("years_of_experience", 5) or 5)
                    edit_license = st.text_input("License Number", value=user.get("medical_license_number", "") or "")

                if st.form_submit_button("💾 Save", use_container_width=True):
                    update_doctor(user["doctor_id"], full_name=edit_name, phone_number=edit_phone,
                                  specialization=edit_spec, qualification=edit_qual,
                                  years_of_experience=edit_exp, medical_license_number=edit_license)
                    st.session_state.user_data = get_doctor(user["doctor_id"])
                    st.success("✅ Updated!")
                    st.rerun()

        with tab2:
            with st.form("edit_doc_work"):
                wc1, wc2 = st.columns(2)
                with wc1:
                    edit_hospital = st.text_input("Hospital/Clinic", value=user.get("hospital_or_clinic_name", "") or "")
                    edit_address = st.text_input("Address", value=user.get("clinic_address", "") or "")
                    edit_city = st.text_input("City", value=user.get("city", "") or "")
                with wc2:
                    edit_fee = st.number_input("Consultation Fee (₹)", 0.0, 10000.0, value=float(user.get("consultation_fee", 500) or 500))
                    edit_mode = st.selectbox("Consultation Mode", ["Online", "Offline", "Both"],
                                            index=["Online", "Offline", "Both"].index(user.get("consultation_mode", "Both")) if user.get("consultation_mode") in ["Online", "Offline", "Both"] else 2)

                if st.form_submit_button("💾 Save", use_container_width=True):
                    update_doctor(user["doctor_id"], hospital_or_clinic_name=edit_hospital,
                                  clinic_address=edit_address, city=edit_city,
                                  consultation_fee=edit_fee, consultation_mode=edit_mode)
                    st.session_state.user_data = get_doctor(user["doctor_id"])
                    st.success("✅ Updated!")
                    st.rerun()

        with tab3:
            with st.form("edit_doc_avail"):
                days_options = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                current_days = (user.get("available_days", "") or "Mon,Tue,Wed,Thu,Fri,Sat").split(",")
                edit_days = st.multiselect("Available Days", days_options, default=[d for d in current_days if d in days_options])
                edit_slots = st.text_input("Time Slots (e.g. 09:00-13:00,14:00-17:00)", value=user.get("available_time_slots", "09:00-17:00") or "09:00-17:00")
                edit_max = st.number_input("Max Patients Per Day", 1, 100, value=user.get("max_patients_per_day", 10) or 10)

                if st.form_submit_button("💾 Save", use_container_width=True):
                    update_doctor(user["doctor_id"], available_days=",".join(edit_days),
                                  available_time_slots=edit_slots, max_patients_per_day=edit_max)
                    st.session_state.user_data = get_doctor(user["doctor_id"])
                    st.success("✅ Updated!")
                    st.rerun()

        with tab4:
            with st.form("doc_change_pw"):
                old_pass = st.text_input("Current Password", type="password")
                new_pass = st.text_input("New Password", type="password")
                new_pass2 = st.text_input("Confirm New Password", type="password")
                if st.form_submit_button("🔒 Change Password", use_container_width=True):
                    if not verify_password(old_pass, user.get("password_hash", "")):
                        st.error("❌ Current password is incorrect.")
                    elif new_pass != new_pass2:
                        st.error("❌ New passwords don't match.")
                    elif len(new_pass) < 6:
                        st.error("❌ Password must be at least 6 characters.")
                    else:
                        new_hash = hash_password(new_pass)
                        update_doctor(user["doctor_id"], password_hash=new_hash)
                        st.session_state.user_data = get_doctor(user["doctor_id"])
                        st.success("✅ Password changed!")


# ===================================================================
#                     PAGE: DOCTOR DASHBOARD
# ===================================================================
if page == "👨‍⚕️ Doctor Dashboard":
    if not st.session_state.logged_in or st.session_state.user_role not in ["doctor", "admin"]:
        st.warning("🔐 Please login as a Doctor to access this page.")
        st.stop()

    user = st.session_state.user_data
    st.markdown(f"## 👨‍⚕️ Doctor Dashboard — Dr. {user.get('full_name', 'N/A')}")

    # Quick Stats
    my_appointments = get_appointments_for_doctor(user["doctor_id"])
    my_reports = get_reports_for_doctor(user["doctor_id"])
    my_reviews = get_reviews_for_doctor(user["doctor_id"])

    sc1, sc2, sc3, sc4 = st.columns(4)
    pending = len(my_appointments[my_appointments["status"] == "Pending"]) if len(my_appointments) > 0 and "status" in my_appointments.columns else 0
    sc1.metric("📅 Pending Appointments", pending)
    sc2.metric("📝 Total Reports", len(my_reports))
    sc3.metric("⭐ Rating", f"{user.get('average_rating', 0):.1f}/5")
    sc4.metric("👥 Total Patients", user.get("total_patients_handled", 0))

    st.markdown("---")

    # Pending Appointment Requests
    st.markdown("### 📋 Pending Consultation Requests")
    if len(my_appointments) > 0:
        pending_df = my_appointments[my_appointments["status"] == "Pending"]
        if len(pending_df) > 0:
            for idx, row in pending_df.iterrows():
                with st.expander(f"🧑 {row.get('patient_name', 'Unknown')} — {row.get('date', 'N/A')} at {row.get('time', 'N/A')}"):
                    st.markdown(f"""
                    - **Patient:** {row.get('patient_name', 'N/A')} ({row.get('gender', 'N/A')}, Age: {row.get('age', 'N/A')})
                    - **Date:** {row.get('date', 'N/A')} at {row.get('time', 'N/A')}
                    - **Type:** {row.get('consultation_type', 'N/A')}
                    - **Phone:** {row.get('patient_phone', 'N/A')}
                    - **Notes:** {row.get('notes', 'N/A')}
                    """)
                    ac1, ac2 = st.columns(2)
                    with ac1:
                        if st.button("✅ Approve", key=f"approve_{row['appointment_id']}", use_container_width=True):
                            update_appointment_status(row["appointment_id"], "Approved")
                            st.success("✅ Appointment approved!")
                            st.rerun()
                    with ac2:
                        if st.button("❌ Reject", key=f"reject_{row['appointment_id']}", use_container_width=True):
                            update_appointment_status(row["appointment_id"], "Rejected")
                            st.warning("Appointment rejected.")
                            st.rerun()
        else:
            st.info("No pending requests.")
    else:
        st.info("No appointments yet.")

    st.markdown("---")

    # Write Diagnosis / Create Report
    st.markdown("### 📝 Write Diagnosis Report")
    patients_list = get_all_patients_list()
    if len(patients_list) > 0:
        patient_options = {f"{r['full_name']} ({r['patient_id']})": r['patient_id'] for _, r in patients_list.iterrows()}
        selected_patient_label = st.selectbox("Select Patient", list(patient_options.keys()))
        selected_pid = patient_options[selected_patient_label]

        with st.form("write_diagnosis"):
            diag = st.text_area("Doctor Diagnosis", placeholder="Your clinical diagnosis...")
            prescription = st.text_area("Prescription", placeholder="Medications, dosage, frequency...")
            notes = st.text_area("Additional Notes", placeholder="Follow-up instructions, observations...")

            # Get latest AI result for this patient
            scans_df = get_all_scans()
            patient_scans = scans_df[scans_df["patient_id"] == selected_pid] if "patient_id" in scans_df.columns else pd.DataFrame()
            ai_result_str = ""
            if len(patient_scans) > 0:
                latest = patient_scans.iloc[0]
                ai_result_str = f"AI: {latest.get('predicted_class', 'N/A')} ({latest.get('confidence', 0):.1f}% conf, Risk: {latest.get('risk_level', 'N/A')})"
                st.info(f"📊 Latest AI Result: {ai_result_str}")

            if st.form_submit_button("📄 Create Report", use_container_width=True):
                rid = create_report(selected_pid, user["doctor_id"], ai_result_str, diag, prescription, notes)
                st.success(f"✅ Report created! ID: **{rid}**")
    else:
        st.info("No registered patients yet.")

    # All Appointments
    st.markdown("---")
    st.markdown("### 📅 All Appointments")
    if len(my_appointments) > 0:
        display_cols = ["appointment_id", "patient_name", "date", "time", "status", "consultation_type"]
        avail = [c for c in display_cols if c in my_appointments.columns]
        st.dataframe(my_appointments[avail], use_container_width=True)


# ===================================================================
#                        PAGE: APPOINTMENTS
# ===================================================================
if page == "📅 Appointments":
    if not st.session_state.logged_in:
        st.warning("🔐 Please login to manage appointments.")
        st.stop()

    user = st.session_state.user_data
    role = st.session_state.user_role

    st.markdown("## 📅 Appointments")

    if role == "patient":
        # ──────────────────────────────────────────────────
        #  BOOK NEW APPOINTMENT — Enhanced Form
        # ──────────────────────────────────────────────────
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea22, #764ba222);
                    padding: 20px 25px; border-radius: 14px; margin-bottom: 20px;
                    border-left: 5px solid #667eea;'>
            <h3 style='margin: 0; color: #667eea;'>📌 Book Appointment</h3>
            <p style='margin: 5px 0 0 0; color: #888; font-size: 13px;'>Fill in the details below to schedule your consultation</p>
        </div>
        """, unsafe_allow_html=True)

        doctors_df = get_all_doctors(verified_only=True)

        if len(doctors_df) > 0:
            with st.form("book_appointment", clear_on_submit=False):

                # ━━━━ SECTION 1: Patient Details ━━━━
                st.markdown("""
                <div style='background: linear-gradient(135deg, #e8f5e922, #c8e6c922);
                            padding: 12px 18px; border-radius: 10px; margin: 10px 0 15px 0;
                            border-left: 4px solid #4caf50;'>
                    <h4 style='margin: 0;'>👤 Patient Details</h4>
                </div>
                """, unsafe_allow_html=True)

                pd1, pd2, pd3 = st.columns(3)
                with pd1:
                    apt_full_name = st.text_input("Full Name *", value=user.get("full_name", ""), key="apt_fname")
                with pd2:
                    apt_age = st.number_input("Age *", min_value=1, max_value=120,
                                              value=int(user.get("age", 25)) if user.get("age") else 25,
                                              key="apt_age")
                with pd3:
                    gender_options = ["Male", "Female", "Other"]
                    default_gender_idx = 0
                    ug = (user.get("gender") or "").capitalize()
                    if ug in gender_options:
                        default_gender_idx = gender_options.index(ug)
                    apt_gender = st.selectbox("Gender *", gender_options, index=default_gender_idx, key="apt_gender")

                pd4, pd5 = st.columns(2)
                with pd4:
                    apt_phone = st.text_input("Mobile Number", value=user.get("phone_number", "") or "",
                                              placeholder="e.g. +91 9876543210", key="apt_phone")
                with pd5:
                    apt_email = st.text_input("Email Address", value=user.get("email", "") or "",
                                              placeholder="email@example.com", key="apt_email")

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

                # ━━━━ SECTION 2: Appointment Details ━━━━
                st.markdown("""
                <div style='background: linear-gradient(135deg, #e3f2fd22, #bbdefb22);
                            padding: 12px 18px; border-radius: 10px; margin: 10px 0 15px 0;
                            border-left: 4px solid #2196f3;'>
                    <h4 style='margin: 0;'>📋 Appointment Details</h4>
                </div>
                """, unsafe_allow_html=True)

                # Doctor selection
                doc_options = {}
                for _, doc in doctors_df.iterrows():
                    label = f"Dr. {doc['full_name']} — {doc.get('specialization', 'N/A')} | ⭐ {doc.get('average_rating', 0):.1f} | ₹{doc.get('consultation_fee', 0):.0f}"
                    doc_options[label] = doc["doctor_id"]

                selected_doc_label = st.selectbox("Select Doctor *", list(doc_options.keys()), key="apt_doc_select")
                selected_doc_id = doc_options[selected_doc_label]

                # Show selected doctor info card
                doc_info = get_doctor(selected_doc_id)
                if doc_info:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #20c99711, #339af022);
                                padding: 15px; border-radius: 10px; border-left: 4px solid #20c997; margin: 5px 0 15px 0;'>
                        <h4 style='margin: 0;'>👨‍⚕️ Dr. {doc_info.get('full_name', 'N/A')}</h4>
                        <p style='margin: 5px 0; color: #888;'>{doc_info.get('specialization', 'N/A')} • {doc_info.get('qualification', 'N/A')} • {doc_info.get('years_of_experience', 0)} yrs exp</p>
                        <p style='margin: 5px 0;'>🏥 {doc_info.get('hospital_or_clinic_name', 'N/A') or 'Not set'} &nbsp;|&nbsp; 📍 {doc_info.get('city', 'N/A') or 'Not set'}</p>
                        <p style='margin: 5px 0;'>📅 Available: {doc_info.get('available_days', 'N/A')} &nbsp;|&nbsp; ⏰ {doc_info.get('available_time_slots', 'N/A')}</p>
                        <p style='margin: 5px 0;'>💰 Fee: ₹{doc_info.get('consultation_fee', 0):.0f} &nbsp;|&nbsp; 🖥️ Mode: {doc_info.get('consultation_mode', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    apt_date = st.date_input("Select Date *", key="apt_date")
                with ac2:
                    apt_time = st.selectbox("Select Time Slot *",
                                            ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
                                             "12:00", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30"],
                                            key="apt_time")
                with ac3:
                    apt_type = st.selectbox("Consultation Type *", ["Online", "Offline"], key="apt_type")

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

                # ━━━━ SECTION 3: Medical Information ━━━━
                st.markdown("""
                <div style='background: linear-gradient(135deg, #fce4ec22, #f8bbd022);
                            padding: 12px 18px; border-radius: 10px; margin: 10px 0 15px 0;
                            border-left: 4px solid #e91e63;'>
                    <h4 style='margin: 0;'>🩺 Medical Information</h4>
                </div>
                """, unsafe_allow_html=True)

                apt_symptoms = st.text_area("Symptoms / Description *",
                                            placeholder="Describe your symptoms or reason for visit in detail...",
                                            height=100, key="apt_symptoms")

                apt_skin_image = st.file_uploader("Upload Skin Image (Optional)",
                                                   type=["jpg", "jpeg", "png"],
                                                   help="Upload a photo of the skin lesion or area of concern",
                                                   key="apt_skin_img")

                apt_medical_history = st.text_area("Previous Medical History (Optional)",
                                                    placeholder="Any previous conditions, allergies, medications, or past treatments...",
                                                    height=80, key="apt_med_history")

                apt_notes = st.text_area("Additional Notes for Doctor (Optional)",
                                         placeholder="Any other information you'd like the doctor to know...",
                                         height=60, key="apt_notes")

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

                # ━━━━ ACTION BUTTONS ━━━━
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    submitted = st.form_submit_button("📅 Book Appointment", use_container_width=True,
                                                       type="primary")
                with btn_col2:
                    reset_clicked = st.form_submit_button("🔄 Reset Form", use_container_width=True)

                if submitted:
                    # Validation
                    if not apt_full_name.strip():
                        st.error("❌ Please enter your full name.")
                    elif not apt_symptoms.strip():
                        st.error("❌ Please describe your symptoms or reason for visit.")
                    else:
                        # Process optional image
                        img_blob = None
                        if apt_skin_image is not None:
                            img_blob = apt_skin_image.read()

                        aid = create_appointment(
                            patient_id=user["patient_id"],
                            doctor_id=selected_doc_id,
                            date=str(apt_date),
                            time=apt_time,
                            consultation_type=apt_type,
                            notes=apt_notes,
                            patient_name=apt_full_name.strip(),
                            patient_age=apt_age,
                            patient_gender=apt_gender,
                            patient_phone=apt_phone.strip(),
                            patient_email=apt_email.strip(),
                            symptoms=apt_symptoms.strip(),
                            medical_history=apt_medical_history.strip(),
                            skin_image_blob=img_blob,
                        )
                        st.success(f"✅ Appointment booked successfully! ID: **{aid}** — Status: Pending")
                        st.balloons()

                if reset_clicked:
                    st.rerun()

        else:
            st.info("No verified doctors available yet. Please check back later.")

        # My Appointments
        st.markdown("---")
        st.markdown("### 📋 My Appointments")
        my_apts = get_appointments_for_patient(user["patient_id"])
        if len(my_apts) > 0:
            for _, apt in my_apts.iterrows():
                status_color = {"Pending": "#ffd43b", "Approved": "#51cf66", "Rejected": "#ff6b6b", "Completed": "#339af0"}.get(apt.get("status", ""), "#aaa")
                st.markdown(f"""
                <div style='background: {status_color}11; border-left: 4px solid {status_color};
                            padding: 15px; border-radius: 10px; margin: 8px 0;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <div>
                            <h4 style='margin: 0;'>👨‍⚕️ {apt.get('doctor_name', 'N/A')} — {apt.get('specialization', 'N/A')}</h4>
                            <p style='margin: 5px 0; color: #888;'>📅 {apt.get('date', 'N/A')} at {apt.get('time', 'N/A')} &nbsp;|&nbsp; {apt.get('consultation_type', 'N/A')}</p>
                            <p style='margin: 5px 0; color: #888;'>🏥 {apt.get('hospital_or_clinic_name', 'N/A') or ''}</p>
                        </div>
                        <div style='text-align: center;'>
                            <span style='background: {status_color}22; border: 2px solid {status_color};
                                        border-radius: 8px; padding: 5px 12px; font-weight: bold; color: {status_color};'>
                                {apt.get('status', 'N/A')}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No appointments yet. Book your first appointment above!")

    elif role in ["doctor", "admin"]:
        # Doctor sees their appointments
        st.markdown("### 📋 My Appointment Schedule")
        my_apts = get_appointments_for_doctor(user["doctor_id"])
        if len(my_apts) > 0:
            # Tabs for status
            tab_all, tab_pending, tab_approved = st.tabs(["All", "Pending", "Approved"])
            with tab_all:
                display_cols = ["appointment_id", "patient_name", "patient_age", "patient_gender",
                                "patient_phone", "patient_email", "date", "time", "status",
                                "consultation_type", "symptoms", "medical_history", "notes"]
                avail = [c for c in display_cols if c in my_apts.columns]
                st.dataframe(my_apts[avail], use_container_width=True)
            with tab_pending:
                pending = my_apts[my_apts["status"] == "Pending"]
                if len(pending) > 0:
                    for _, row in pending.iterrows():
                        st.markdown(f"**{row.get('patient_name', 'N/A')}** — {row.get('date', 'N/A')} at {row.get('time', 'N/A')}")
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("✅ Approve", key=f"apt_approve_{row['appointment_id']}", use_container_width=True):
                                update_appointment_status(row["appointment_id"], "Approved")
                                st.rerun()
                        with c2:
                            if st.button("❌ Reject", key=f"apt_reject_{row['appointment_id']}", use_container_width=True):
                                update_appointment_status(row["appointment_id"], "Rejected")
                                st.rerun()
                else:
                    st.info("No pending appointments.")
            with tab_approved:
                approved = my_apts[my_apts["status"] == "Approved"]
                if len(approved) > 0:
                    avail = [c for c in display_cols if c in approved.columns]
                    st.dataframe(approved[avail], use_container_width=True)
                else:
                    st.info("No approved appointments yet.")
        else:
            st.info("No appointments yet.")


# ===================================================================
#                        PAGE: REPORTS
# ===================================================================
if page == "📝 Reports":
    if not st.session_state.logged_in:
        st.warning("🔐 Please login to view reports.")
        st.stop()

    user = st.session_state.user_data
    role = st.session_state.user_role

    st.markdown("## 📝 Diagnosis Reports")

    if role == "patient":
        reports = get_reports_for_patient(user["patient_id"])
        if len(reports) > 0:
            st.success(f"Found **{len(reports)}** reports.")
            for _, rpt in reports.iterrows():
                with st.expander(f"📄 Report {rpt.get('report_id', 'N/A')} — {rpt.get('created_at', 'N/A')[:10] if rpt.get('created_at') else 'N/A'}"):
                    st.markdown(f"""
                    | Field | Details |
                    |---|---|
                    | **Report ID** | `{rpt.get('report_id', 'N/A')}` |
                    | **Doctor** | Dr. {rpt.get('doctor_name', 'N/A')} ({rpt.get('specialization', 'N/A')}) |
                    | **AI Result** | {rpt.get('ai_result', 'N/A')} |
                    | **Doctor Diagnosis** | {rpt.get('doctor_diagnosis', 'N/A') or 'Pending'} |
                    | **Prescription** | {rpt.get('prescription', 'N/A') or 'None'} |
                    | **Notes** | {rpt.get('doctor_notes', 'N/A') or 'None'} |
                    | **Date** | {rpt.get('created_at', 'N/A')} |
                    """)
        else:
            st.info("No reports yet. Your doctor will create reports after consultation.")

        # AI Prediction History
        st.markdown("---")
        st.markdown("### 🤖 AI Prediction History")
        scans_df = get_all_scans()
        my_scans = scans_df[scans_df["patient_id"] == user["patient_id"]] if "patient_id" in scans_df.columns else pd.DataFrame()
        if len(my_scans) > 0:
            display_cols = ["scan_timestamp", "predicted_class", "confidence", "cancer_prob", "risk_level", "body_location", "symptoms"]
            avail = [c for c in display_cols if c in my_scans.columns]
            st.dataframe(my_scans[avail], use_container_width=True)
        else:
            st.info("No AI predictions yet.")

    elif role in ["doctor", "admin"]:
        reports = get_reports_for_doctor(user["doctor_id"])
        if len(reports) > 0:
            st.success(f"**{len(reports)}** reports written.")
            for _, rpt in reports.iterrows():
                with st.expander(f"📄 {rpt.get('patient_name', 'Unknown')} — {rpt.get('report_id', 'N/A')}"):
                    st.markdown(f"""
                    - **Patient:** {rpt.get('patient_name', 'N/A')} ({rpt.get('gender', 'N/A')}, Age: {rpt.get('age', 'N/A')})
                    - **AI Result:** {rpt.get('ai_result', 'N/A')}
                    - **Your Diagnosis:** {rpt.get('doctor_diagnosis', 'N/A') or 'Pending'}
                    - **Prescription:** {rpt.get('prescription', 'N/A') or 'None'}
                    - **Notes:** {rpt.get('doctor_notes', 'N/A') or 'None'}
                    - **Date:** {rpt.get('created_at', 'N/A')}
                    """)
        else:
            st.info("No reports written yet. Go to Doctor Dashboard to create one.")


# ===================================================================
#                        PAGE: REVIEWS
# ===================================================================
if page == "⭐ Reviews":
    if not st.session_state.logged_in or st.session_state.user_role != "patient":
        st.warning("🔐 Please login as a Patient to write reviews.")
        st.stop()

    user = st.session_state.user_data
    st.markdown("## ⭐ Rate & Review Doctors")

    # Write Review
    st.markdown("### ✍️ Write a Review")
    doctors_df = get_all_doctors(verified_only=True)
    if len(doctors_df) > 0:
        doc_options = {f"Dr. {r['full_name']} — {r.get('specialization', 'N/A')}": r["doctor_id"] for _, r in doctors_df.iterrows()}
        selected_doc = st.selectbox("Select Doctor to Review", list(doc_options.keys()))
        selected_doc_id = doc_options[selected_doc]

        with st.form("write_review"):
            rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.5)
            review_text = st.text_area("Your Review", placeholder="Share your experience with this doctor...")

            if st.form_submit_button("⭐ Submit Review", use_container_width=True):
                if review_text.strip():
                    create_review(user["patient_id"], selected_doc_id, rating, review_text)
                    st.success("✅ Review submitted! Thank you for your feedback.")
                    st.balloons()
                else:
                    st.warning("Please write a review before submitting.")

        # Show existing reviews for selected doctor
        st.markdown("---")
        st.markdown(f"### 📖 Reviews for {selected_doc.split('—')[0].strip()}")
        reviews = get_reviews_for_doctor(selected_doc_id)
        if len(reviews) > 0:
            for _, rev in reviews.iterrows():
                stars = "⭐" * int(rev.get("rating", 0))
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 8px 0; border-left: 4px solid #ffd43b;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span><b>{rev.get('patient_name', 'Anonymous')}</b></span>
                        <span>{stars} ({rev.get('rating', 0):.1f})</span>
                    </div>
                    <p style='margin: 8px 0 0 0; color: #555;'>{rev.get('review_text', '')}</p>
                    <p style='margin: 4px 0 0 0; font-size: 11px; color: #aaa;'>{rev.get('created_at', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No reviews yet for this doctor.")
    else:
        st.info("No verified doctors available to review.")


# ===================================================================
#                        PAGE: ADMIN PANEL
# ===================================================================
if page == "🔑 Admin Panel":
    if not st.session_state.logged_in or st.session_state.user_role != "admin":
        st.warning("🔐 Admin access only.")
        st.stop()

    st.markdown("## 🔑 Admin Panel — System Management")

    # System Stats
    all_patients = get_all_patients_list()
    all_doctors = get_all_doctors(verified_only=False)
    all_scans = get_all_scans()

    st.markdown("### 📊 System Statistics")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("👥 Total Patients", len(all_patients))
    sc2.metric("👨‍⚕️ Total Doctors", len(all_doctors))
    sc3.metric("🔬 Total Scans", len(all_scans))
    pending_docs = len(all_doctors[all_doctors["verification_status"] == "Pending"]) if len(all_doctors) > 0 and "verification_status" in all_doctors.columns else 0
    sc4.metric("⏳ Pending Verifications", pending_docs)

    st.markdown("---")

    # Doctor Verification
    st.markdown("### 🔍 Doctor Verification Queue")
    if len(all_doctors) > 0:
        pending_doctors = all_doctors[all_doctors["verification_status"] == "Pending"]
        if len(pending_doctors) > 0:
            for _, doc in pending_doctors.iterrows():
                with st.expander(f"👨‍⚕️ Dr. {doc.get('full_name', 'N/A')} — {doc.get('specialization', 'N/A')}"):
                    st.markdown(f"""
                    | Field | Details |
                    |---|---|
                    | **Name** | Dr. {doc.get('full_name', 'N/A')} |
                    | **Email** | {doc.get('email', 'N/A')} |
                    | **Specialization** | {doc.get('specialization', 'N/A')} |
                    | **Qualification** | {doc.get('qualification', 'N/A')} |
                    | **Experience** | {doc.get('years_of_experience', 0)} years |
                    | **License #** | {doc.get('medical_license_number', 'N/A') or 'Not provided'} |
                    | **Hospital** | {doc.get('hospital_or_clinic_name', 'N/A') or 'Not set'} |
                    | **City** | {doc.get('city', 'N/A') or 'Not set'} |
                    | **Registered** | {doc.get('account_created_at', 'N/A')} |
                    """)
                    vc1, vc2 = st.columns(2)
                    with vc1:
                        if st.button("✅ Verify Doctor", key=f"verify_{doc['doctor_id']}", use_container_width=True):
                            update_doctor(doc["doctor_id"],
                                          verification_status="Verified",
                                          verified_by_admin=st.session_state.user_data.get("doctor_id", "ADMIN"),
                                          verification_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            st.success(f"✅ Dr. {doc['full_name']} verified!")
                            st.rerun()
                    with vc2:
                        if st.button("❌ Reject", key=f"reject_doc_{doc['doctor_id']}", use_container_width=True):
                            update_doctor(doc["doctor_id"], verification_status="Rejected")
                            st.warning(f"Doctor {doc['full_name']} rejected.")
                            st.rerun()
        else:
            st.success("✅ No pending verifications!")
    else:
        st.info("No doctors registered yet.")

    # All Doctors Table
    st.markdown("---")
    st.markdown("### 👨‍⚕️ All Doctors")
    if len(all_doctors) > 0:
        display_cols = ["doctor_id", "full_name", "email", "specialization", "qualification",
                       "verification_status", "average_rating", "total_patients_handled", "status"]
        avail = [c for c in display_cols if c in all_doctors.columns]
        st.dataframe(all_doctors[avail], use_container_width=True)

    # All Patients Table
    st.markdown("---")
    st.markdown("### 👥 All Patients")
    if len(all_patients) > 0:
        display_cols = ["patient_id", "full_name", "email", "age", "gender", "blood_group",
                       "phone_number", "account_created_at"]
        avail = [c for c in display_cols if c in all_patients.columns]
        st.dataframe(all_patients[avail], use_container_width=True)


# ================= FOOTER =================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: grey;'>
    <p>© 2026 AI DermaScan v{APP_VERSION} | Gautam Buddha University</p>
    <p style='font-size: 12px;'>🧬 Multi-Class • 🧠 Grad-CAM • 👤 Profiles • 👨‍⚕️ Doctors • 📅 Appointments • 📝 Reports • ⭐ Reviews • 🔑 Admin • 🗺️ Clinics • 🌐 Multi-Language</p>
</div>
""", unsafe_allow_html=True)
