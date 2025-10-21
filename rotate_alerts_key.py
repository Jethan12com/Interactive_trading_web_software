"""
rotate_alerts_key.py
----------------------------------------------------
Safely rotates the encryption key for AlertManager’s
encrypted alert cache (alert_cache.json.enc).

✅ Decrypts existing cache with the old key
✅ Generates and saves a new Fernet key
✅ Re-encrypts the cache with the new key
✅ Creates a timestamped backup of the old cache/key

Usage:
    python rotate_alerts_key.py
"""

import os
import json
import shutil
from datetime import datetime
from cryptography.fernet import Fernet, InvalidToken

# === File Paths (match AlertManager) ===
SECRETS_DIR = "secrets"
LOGS_DIR = "logs"
KEY_FILE = os.path.join(SECRETS_DIR, "alerts_key.key")
CACHE_FILE = os.path.join(LOGS_DIR, "alert_cache.json.enc")
BACKUP_DIR = os.path.join(LOGS_DIR, "cache_backups")

os.makedirs(SECRETS_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)


def rotate_key():
    print("\n🔁 Starting secure key rotation for AlertManager...\n")

    if not os.path.exists(KEY_FILE):
        print("⚠️  No key file found. Nothing to rotate.")
        return

    if not os.path.exists(CACHE_FILE):
        print("⚠️  No encrypted cache found. Nothing to re-encrypt.")
        return

    # === Step 1: Load old key ===
    try:
        with open(KEY_FILE, "rb") as f:
            old_key = f.read().strip()
        fernet_old = Fernet(old_key)
        print("🔑 Old encryption key loaded.")
    except Exception as e:
        print(f"❌ Failed to load old key: {e}")
        return

    # === Step 2: Backup old files ===
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    key_backup = os.path.join(BACKUP_DIR, f"alerts_key_{ts}.key.bak")
    cache_backup = os.path.join(BACKUP_DIR, f"alert_cache_{ts}.enc.bak")

    try:
        shutil.copy2(KEY_FILE, key_backup)
        shutil.copy2(CACHE_FILE, cache_backup)
        print(f"🧾 Backed up old key and cache to {BACKUP_DIR}")
    except Exception as e:
        print(f"⚠️ Backup failed: {e}")

    # === Step 3: Decrypt cache ===
    try:
        with open(CACHE_FILE, "rb") as f:
            encrypted_data = f.read()
        decrypted = fernet_old.decrypt(encrypted_data)
        data = json.loads(decrypted.decode("utf-8"))
        print(f"📂 Decrypted existing cache with {len(data)} entries.")
    except InvalidToken:
        print("❌ Invalid decryption key or corrupted cache file.")
        return
    except Exception as e:
        print(f"❌ Cache decryption failed: {e}")
        return

    # === Step 4: Generate new key ===
    new_key = Fernet.generate_key()
    fernet_new = Fernet(new_key)
    try:
        with open(KEY_FILE, "wb") as f:
            f.write(new_key)
        os.chmod(KEY_FILE, 0o600)
        print(f"🔐 New encryption key generated and saved at {KEY_FILE}")
    except Exception as e:
        print(f"❌ Failed to write new key: {e}")
        return

    # === Step 5: Re-encrypt and save cache ===
    try:
        encrypted_new = fernet_new.encrypt(json.dumps(data, indent=2).encode("utf-8"))
        tmp_path = CACHE_FILE + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(encrypted_new)
        os.replace(tmp_path, CACHE_FILE)
        print("✅ Cache successfully re-encrypted with new key.")
    except Exception as e:
        print(f"❌ Failed to re-encrypt cache: {e}")
        return

    print("\n🎉 Key rotation completed successfully.")
    print(f"📦 Backups stored in: {BACKUP_DIR}\n")


if __name__ == "__main__":
    rotate_key()