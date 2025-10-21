import os
import sys
import pandas as pd
import getpass
import hashlib
from tabulate import tabulate
from modules.encryption_helper import encrypt_text, decrypt_text

DATA_DIR = "data"
ADMIN_FILE = os.path.join(DATA_DIR, "admins.csv")


# =====================================================
# Helpers
# =====================================================
def hash_password(password: str) -> str:
    """Hash a password with SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def ensure_admin_file():
    """Ensure admin file exists."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ADMIN_FILE):
        df = pd.DataFrame(columns=["username", "password_hash", "telegram_id", "role"])
        df.to_csv(ADMIN_FILE, index=False)
        print(f"[INFO] Created admin file at {ADMIN_FILE}")


def load_admins():
    ensure_admin_file()
    try:
        return pd.read_csv(ADMIN_FILE)
    except Exception as e:
        print(f"[ERROR] Failed to read admin file: {e}")
        sys.exit(1)


def save_admins(df):
    try:
        df.to_csv(ADMIN_FILE, index=False)
        print("[INFO] Admin file updated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to write admin file: {e}")
        sys.exit(1)


# =====================================================
# Core Commands
# =====================================================
def create_admin():
    df = load_admins()

    print("=== Create New Admin ===")
    username = input("Enter username: ").strip()
    if username in df["username"].values:
        print(f"[ERROR] Admin '{username}' already exists.")
        return

    password = getpass.getpass("Enter password: ").strip()
    confirm = getpass.getpass("Confirm password: ").strip()
    if password != confirm:
        print("[ERROR] Passwords do not match.")
        return

    telegram_id = input("Enter Telegram ID (optional): ").strip()
    role = input("Enter role (default: admin): ").strip() or "admin"

    hashed = hash_password(password)
    encrypted = encrypt_text(hashed)

    new_admin = {
        "username": username,
        "password_hash": encrypted,
        "telegram_id": telegram_id,
        "role": role,
    }

    df = pd.concat([df, pd.DataFrame([new_admin])], ignore_index=True)
    save_admins(df)

    print(f"[SUCCESS] Admin '{username}' created successfully.")


def list_admins():
    df = load_admins()
    if df.empty:
        print("[INFO] No admins found.")
        return

    df_display = df.copy()
    df_display["password_hash"] = df_display["password_hash"].apply(lambda x: x[:10] + "..." if isinstance(x, str) else "")
    print("\n=== Registered Admins ===")
    print(tabulate(df_display, headers="keys", tablefmt="grid"))


def delete_admin():
    df = load_admins()
    username = input("Enter username to delete: ").strip()

    if username not in df["username"].values:
        print(f"[ERROR] No admin found with username '{username}'.")
        return

    confirm = input(f"Are you sure you want to delete '{username}'? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("[INFO] Deletion cancelled.")
        return

    df = df[df["username"] != username]
    save_admins(df)
    print(f"[SUCCESS] Admin '{username}' deleted.")


# =====================================================
# CLI Menu
# =====================================================
def main():
    ensure_admin_file()

    print("""
===================================
   CoPilot Admin Management CLI
===================================
1. Create Admin
2. List Admins
3. Delete Admin
4. Exit
""")

    while True:
        choice = input("Select option [1-4]: ").strip()

        if choice == "1":
            create_admin()
        elif choice == "2":
            list_admins()
        elif choice == "3":
            delete_admin()
        elif choice == "4":
            print("[EXIT] Goodbye.")
            sys.exit(0)
        else:
            print("[ERROR] Invalid option. Please try again.")

        input("\nPress Enter to continue...")
        os.system("cls" if os.name == "nt" else "clear")
        main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] Operation cancelled by user.")
