import pandas as pd
import os

# Path to users.csv in project root
users_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users.csv")

# Pre-filled users
users_data = [
    {"user_id":"U1", "name":"Michael", "telegram_id":"@michael123", "preferred_session":"New York"},
    {"user_id":"U2", "name":"John", "telegram_id":"@john456", "preferred_session":"Tokyo"},
    {"user_id":"U3", "name":"Peace", "telegram_id":"@peace789", "preferred_session":"London"},
    {"user_id":"U4", "name":"Helen", "telegram_id":"@helen101", "preferred_session":"Sydney"}
]

# Save to CSV
df = pd.DataFrame(users_data)
df.to_csv(users_file, index=False)
print(f"✅ users.csv created at {users_file}")
