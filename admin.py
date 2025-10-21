import os
import pandas as pd
import hashlib
from cryptography.fernet import Fernet

# -----------------------------
# CONFIG
# -----------------------------
ADMIN_FILE = "data/admins.csv"
KEY_FILE = "data/secret.key"

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def generate_key():
    """Create a key if none exists."""
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    else:
        with open(KEY_FILE, "rb") as f:
            key = f.read()
    return key

def encrypt_text(text: str, key: bytes) -> str:
    f = Fernet(key)
    return f.encrypt(text.encode()).decode()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# -----------------------------
# CREATE ADMIN
# -----------------------------
key = generate_key()

username = "admin"           # change if desired
password = "StrongPass!23"   # change if desired
telegram_id = ""             # optional
role = "admin"

# Create CSV if missing
if not os.path.exists(ADMIN_FILE):
    pd.DataFrame(columns=["username", "password_hash", "telegram_id", "role"]).to_csv(ADMIN_FILE, index=False)

df = pd.read_csv(ADMIN_FILE)

if username in df["username"].values:
    print(f"⚠️ Admin '{username}' already exists. Updating password.")
    df.loc[df["username"] == username, "password_hash"] = encrypt_text(hash_password(password), key)
else:
    new_row = {
        "username": username,
        "password_hash": encrypt_text(hash_password(password), key),
        "telegram_id": telegram_id,
        "role": role
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df.to_csv(ADMIN_FILE, index=False)
print(f"✅ Admin '{username}' created/updated successfully!")
print(f"🔐 Password: {password} (store safely)")