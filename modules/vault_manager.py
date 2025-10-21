import hvac
import logging
import os
from datetime import datetime
import pytz
import time
from requests import Session
from requests.exceptions import ConnectionError
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Logging setup
logger = logging.getLogger("VaultSecretsManager")
logger.setLevel(logging.INFO)
os.makedirs('logs', exist_ok=True)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z"
)
formatter.converter = lambda *args: datetime.now(
    pytz.timezone('Africa/Lagos')
).timetuple()
file_handler = logging.FileHandler(
    os.path.join('logs', 'vault_secrets_manager.log'), encoding="utf-8"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class VaultSecretsManager:
    def __init__(self, vault_url: str, vault_token: str, retries: int = 3, backoff_factor: float = 1.0):
        logger.info(f"Initializing Vault client with URL: {vault_url}")
        self.client = None
        # Configure retries with exponential backoff
        session = Session()
        retries_config = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False
        )
        session.mount("http://", HTTPAdapter(max_retries=retries_config))
        session.mount("https://", HTTPAdapter(max_retries=retries_config))
        self.client = hvac.Client(url=vault_url, token=vault_token, session=session)
        
        # Retry authentication
        for attempt in range(retries + 1):
            try:
                if self.client.is_authenticated():
                    logger.info("Vault client authenticated successfully")
                    break
                else:
                    logger.error("Vault authentication failed")
                    raise Exception("Invalid Vault token or unreachable server")
            except (ConnectionError, Exception) as e:
                if attempt < retries:
                    sleep_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Vault connection attempt {attempt + 1}/{retries + 1} failed: {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Vault initialization failed after {retries + 1} attempts: {e}. Proceeding without Vault secrets.")
                    self.client = None

    def get_api_keys(self, secret_name: str, retries: int = 3, backoff_factor: float = 1.0) -> list:
        """Retrieve API keys from Vault with retry logic."""
        if not self.client:
            logger.warning(f"No Vault client available, cannot fetch {secret_name}")
            return []
        for attempt in range(retries + 1):
            try:
                response = self.client.secrets.kv.v2.read_secret_version(path=secret_name)
                keys = response['data']['data'].get('keys', [])
                logger.info(f"Retrieved API keys for {secret_name}")
                return keys
            except (ConnectionError, Exception) as e:
                if attempt < retries:
                    sleep_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Failed to retrieve API keys for {secret_name}, attempt {attempt + 1}/{retries + 1}: {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to retrieve API keys for {secret_name} after {retries + 1} attempts: {e}")
                    return []

    def rotate_key(self, secret_name: str, index: int, new_key: dict) -> bool:
        """Rotate a specific API key in Vault."""
        if not self.client:
            logger.warning(f"No Vault client, cannot rotate key for {secret_name}")
            return False
        try:
            keys = self.get_api_keys(secret_name)
            if index < len(keys):
                keys[index] = new_key
                self.client.secrets.kv.v2.create_or_update_secret(path=secret_name, secret={'keys': keys})
                logger.info(f"Rotated key {index} for {secret_name}")
                return True
            else:
                logger.warning(f"Invalid key index {index} for {secret_name}")
                return False
        except Exception as e:
            logger.error(f"Failed to rotate key {index} for {secret_name}: {e}")
            return False

    def remove_old_key(self, secret_name: str, index: int) -> None:
        """Remove an old API key from Vault."""
        if not self.client:
            logger.warning(f"No Vault client, cannot remove key for {secret_name}")
            return
        try:
            keys = self.get_api_keys(secret_name)
            if index < len(keys):
                del keys[index]
                self.client.secrets.kv.v2.create_or_update_secret(path=secret_name, secret={'keys': keys})
                logger.info(f"Removed old key {index} for {secret_name}")
            else:
                logger.warning(f"Invalid key index {index} for {secret_name}")
        except Exception as e:
            logger.error(f"Failed to remove old key {index} for {secret_name}: {e}")

    def add_new_key(self, secret_name: str, api_key: str, api_secret: str) -> bool:
        """Add a new API key to Vault."""
        if not self.client:
            logger.warning(f"No Vault client, cannot add key for {secret_name}")
            return False
        try:
            keys = self.get_api_keys(secret_name)
            new_key = {
                'api_key': api_key,
                'api_secret': api_secret,
                'creation_date': datetime.now(pytz.timezone('Africa/Lagos')).isoformat()
            }
            keys.append(new_key)
            self.client.secrets.kv.v2.create_or_update_secret(path=secret_name, secret={'keys': keys})
            logger.info(f"Added new key for {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add new key for {secret_name}: {e}")
            return False

    def list_secrets(self) -> list:
        """List all secrets in Vault."""
        if not self.client:
            logger.warning("No Vault client available, cannot list secrets")
            return []
        try:
            response = self.client.secrets.kv.v2.list_secrets(path='')
            secrets = response['data']['keys']
            logger.info(f"Listed secrets: {secrets}")
            return secrets
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []