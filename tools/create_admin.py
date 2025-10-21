import sys
import os
import getpass
import pandas as pd
import hashlib
from cryptography.fernet import Fernet
import secrets
import string

# Ensure base directory is in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# -----------------------------
# FILE PATHS
# -----------------------------
ADMIN_FILE = "data/admins.csv"
KEY_FILE = "data/secret.key"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def generate_key():
    """Create or load a Fernet key for encryption."""
    os.makedirs("data", exist_ok=True)
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

def generate_password(length=14):
    """Generate a secure random password."""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def show_recovery_command(username, password):
    print("\n🛠️  Recovery Command (use this if you ever need to recreate the admin):")
    print("────────────────────────────────────────────────────────────")
    print(f"python -m copilot.tools.create_admin --auto \"{username}\" \"{password}\"")
    print("────────────────────────────────────────────────────────────\n")

# -----------------------------
# CREATE OR RESET ADMIN
# -----------------------------
def create_or_reset_admin(username, password, telegram_id=""):
    key = generate_key()

    os.makedirs("data", exist_ok=True)
    if not os.path.exists(ADMIN_FILE):
        pd.DataFrame(columns=["username", "password_hash", "telegram_id", "role"]).to_csv(ADMIN_FILE, index=False)

    df = pd.read_csv(ADMIN_FILE)

    encrypted_hash = encrypt_text(hash_password(password), key)

    if username in df["username"].values:
        df.loc[df["username"] == username, "password_hash"] = encrypted_hash
        df.loc[df["username"] == username, "telegram_id"] = telegram_id
        action = "updated"
    else:
        new_row = {
            "username": username,
            "password_hash": encrypted_hash,
            "telegram_id": telegram_id,
            "role": "admin"
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        action = "created"

    df.to_csv(ADMIN_FILE, index=False)
    print(f"✅ Admin '{username}' {action} successfully!")
    print(f"🔐 Password: {password} (store safely)")
    show_recovery_command(username, password)

# -----------------------------
# CLI INTERFACE
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        if len(sys.argv) != 4:
            print("Usage: python -m copilot.tools.create_admin --auto <username> <password>")
            sys.exit(1)
        _, _, username, password = sys.argv
        create_or_reset_admin(username, password)

    else:
        print("\n🔐 CoPilot Admin Account Creator\n")
        username = input("Enter admin username: ").strip()
        password = getpass.getpass("Enter password (or press Enter to auto-generate): ").strip()
        if not password:
            password = generate_password()
            print(f"🔑 Auto-generated secure password: {password}")
        confirm = getpass.getpass("Confirm password (press Enter to accept auto-generated): ").strip() or password
        if password != confirm:
            print("❌ Passwords do not match. Exiting.")
            sys.exit(1)

        telegram_id = input("Enter Telegram ID (optional): ").strip()
        create_or_reset_admin(username, password, telegram_id)