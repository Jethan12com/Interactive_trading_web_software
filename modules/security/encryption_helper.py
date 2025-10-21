from cryptography.fernet import Fernet, MultiFernet
import os
import base64
import hvac
import logging
from datetime import datetime, timedelta
import pytz

# Logger Setup
logger = logging.getLogger("EncryptionHelper")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler("logs/encryption_helper.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Key Management
KEY_FILE = "modules/security/.secret.key"
VAULT_PATH = "secret/data/encryption"

class EncryptionHelper:
    def __init__(self):
        self.vault_url = os.getenv("VAULT_URL", "http://127.0.0.1:8201")
        self.vault_token = os.getenv("VAULT_TOKEN")
        self.client = hvac.Client(url=self.vault_url, token=self.vault_token)
        self.active_keys = {}  # {key_id: fernet_instance}
        self.load_keys()
        self.multi_fernet = MultiFernet(list(self.active_keys.values()))

    def load_keys(self):
        """Load keys from Vault or local file, initialize MultiFernet."""
        try:
            if self.client.is_authenticated():
                response = self.client.secrets.kv.v2.read_secret_version(path="encryption", mount_point="secret")
                keys_data = response["data"]["data"].get("keys", {})
                for key_id, key_info in keys_data.items():
                    key = base64.b64decode(key_info["key"])
                    self.active_keys[key_id] = Fernet(key)
                logger.info(f"Loaded {len(self.active_keys)} keys from Vault")
            else:
                logger.warning("Vault not authenticated, falling back to local key")
                if not os.path.exists(KEY_FILE):
                    self.generate_key("local_key")
                with open(KEY_FILE, "rb") as f:
                    key = f.read()
                    self.active_keys["local_key"] = Fernet(key)
                logger.info("Loaded local key")
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            if not os.path.exists(KEY_FILE):
                self.generate_key("local_key")
            with open(KEY_FILE, "rb") as f:
                key = f.read()
                self.active_keys["local_key"] = Fernet(key)
            logger.info("Initialized with new local key")

    def generate_key(self, key_id: str):
        """Generate a new Fernet key and save to Vault or file."""
        try:
            key = Fernet.generate_key()
            fernet = Fernet(key)
            creation_date = datetime.now(pytz.timezone("Africa/Lagos")).isoformat()
            if self.client.is_authenticated():
                response = self.client.secrets.kv.v2.read_secret_version(path="encryption", mount_point="secret")
                keys_data = response["data"]["data"].get("keys", {}) if response else {}
                keys_data[key_id] = {
                    "key": base64.b64encode(key).decode("utf-8"),
                    "creation_date": creation_date
                }
                self.client.secrets.kv.v2.create_or_update_secret(
                    path="encryption",
                    secret={"keys": keys_data},
                    mount_point="secret"
                )
                logger.info(f"Generated and stored new key {key_id} in Vault")
            else:
                os.makedirs(os.path.dirname(KEY_FILE), exist_ok=True)
                with open(KEY_FILE, "wb") as f:
                    f.write(key)
                logger.info(f"Generated and stored new key {key_id} in {KEY_FILE}")
            self.active_keys[key_id] = fernet
            self.multi_fernet = MultiFernet(list(self.active_keys.values()))
            return key
        except Exception as e:
            logger.error(f"Failed to generate key {key_id}: {e}")
            raise

    def rotate_key(self, new_key_id: str, max_age_days: int = 90):
        """Rotate keys, keeping old keys for decryption, and mark old keys for re-encryption."""
        try:
            # Generate new key
            self.generate_key(new_key_id)
            # Remove keys older than max_age_days
            if self.client.is_authenticated():
                response = self.client.secrets.kv.v2.read_secret_version(path="encryption", mount_point="secret")
                keys_data = response["data"]["data"].get("keys", {})
                current_time = datetime.now(pytz.timezone("Africa/Lagos"))
                updated_keys = {}
                for key_id, key_info in keys_data.items():
                    try:
                        creation_date = datetime.fromisoformat(key_info["creation_date"])
                        if (current_time - creation_date).days <= max_age_days:
                            updated_keys[key_id] = key_info
                        else:
                            logger.info(f"Retiring key {key_id} (older than {max_age_days} days)")
                            del self.active_keys[key_id]
                    except Exception as e:
                        logger.warning(f"Invalid creation_date for key {key_id}: {e}")
                        updated_keys[key_id] = key_info
                self.client.secrets.kv.v2.create_or_update_secret(
                    path="encryption",
                    secret={"keys": updated_keys},
                    mount_point="secret"
                )
            self.multi_fernet = MultiFernet(list(self.active_keys.values()))
            logger.info(f"Key rotation completed, active keys: {list(self.active_keys.keys())}")
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            raise

    def encrypt_text(self, plain_text: str, key_id: str = None) -> tuple[str, str]:
        """Encrypt a string with the latest key, return (cipher_text, key_id)."""
        if not plain_text:
            return "", ""
        try:
            if not key_id or key_id not in self.active_keys:
                key_id = max(self.active_keys.keys(), key=lambda k: k if k != "local_key" else "0")
            cipher_text = self.active_keys[key_id].encrypt(plain_text.encode("utf-8")).decode("utf-8")
            logger.info(f"Encrypted text with key {key_id}")
            return cipher_text, key_id
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_text(self, cipher_text: str, key_id: str = None) -> str:
        """Decrypt a string using the appropriate key."""
        if not cipher_text:
            return ""
        try:
            if key_id and key_id in self.active_keys:
                plain = self.active_keys[key_id].decrypt(cipher_text.encode("utf-8")).decode("utf-8")
            else:
                plain = self.multi_fernet.decrypt(cipher_text.encode("utf-8")).decode("utf-8")
            logger.info(f"Decrypted text with key {key_id or 'MultiFernet'}")
            return plain
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def reencrypt_data(self, old_key_id: str, new_key_id: str, data: list[dict]) -> list[dict]:
        """Re-encrypt a list of dictionaries for specified fields."""
        reencrypted_data = []
        for item in data:
            new_item = item.copy()
            for field in ["telegram_id", "activation_token", "user_id"]:
                if field in new_item and new_item[field]:
                    try:
                        plain = self.decrypt_text(new_item[field], old_key_id)
                        new_item[field], _ = self.encrypt_text(plain, new_key_id)
                    except Exception as e:
                        logger.warning(f"Failed to re-encrypt {field} for item: {e}")
                        continue
            reencrypted_data.append(new_item)
        logger.info(f"Re-encrypted {len(reencrypted_data)} items from key {old_key_id} to {new_key_id}")
        return reencrypted_data

if __name__ == "__main__":
    enc = EncryptionHelper()
    sample = "mypassword123"
    cipher_text, key_id = enc.encrypt_text(sample)
    decrypted = enc.decrypt_text(cipher_text, key_id)
    print(f"Original: {sample}")
    print(f"Encrypted: {cipher_text} (Key ID: {key_id})")
    print(f"Decrypted: {decrypted}")
    enc.rotate_key("key2")
    reencrypted = enc.reencrypt_data("local_key", "key2", [{"telegram_id": cipher_text, "data": "test"}])
    print(f"Re-encrypted: {reencrypted}")