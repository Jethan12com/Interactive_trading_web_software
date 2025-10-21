import os
from cryptography.fernet import Fernet
import logging

KEY_FILE = "config/fernet.key"
os.makedirs("config", exist_ok=True)

logger = logging.getLogger("encryption_helper")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def load_or_create_key():
    """Load existing Fernet key or create one if missing."""
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        logger.info(f"Generated new Fernet key at {KEY_FILE}")
    else:
        with open(KEY_FILE, "rb") as f:
            key = f.read()
    return Fernet(key)

fernet = load_or_create_key()

def encrypt_text(text: str) -> str:
    """Encrypt a string (returns base64)."""
    if text is None:
        return ""
    return fernet.encrypt(text.encode()).decode()

def decrypt_text(encrypted_text: str) -> str:
    """Decrypt a previously encrypted string."""
    if not encrypted_text:
        return ""
    try:
        return fernet.decrypt(encrypted_text.encode()).decode()
    except Exception:
        # backward compatibility for unencrypted old entries
        return encrypted_text