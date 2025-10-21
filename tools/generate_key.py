from cryptography.fernet import Fernet
import os

KEY_PATH = "config/admin_key.key"

os.makedirs("config", exist_ok=True)

if os.path.exists(KEY_PATH):
    print(f"[INFO] Key already exists at {KEY_PATH}")
else:
    key = Fernet.generate_key()
    with open(KEY_PATH, "wb") as f:
        f.write(key)
    print(f"[SUCCESS] Generated new admin encryption key at {KEY_PATH}")
