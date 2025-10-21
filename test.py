from modules.user_management import UserManagement
from datetime import datetime, timedelta
import pytz
import uuid
um = UserManagement()
user_data = {
    "telegram_id": "12345",
    "username": "testuser",
    "tier": "Premium",
    "account_type": "Demo",
    "risk_profile": "Medium",
    "time_zone": "Africa/Lagos",
    "active_sessions": ["London", "NewYork"],
    "capital": 2000.0,
    "preferred_pairs": ["EUR/USD", "GBP/USD"],
    "activation_token": str(uuid.uuid4()),
    "activation_token_expiry": (datetime.now(pytz.timezone('Africa/Lagos')) + timedelta(hours=24)).isoformat(),
    "expiry_date": (datetime.now(pytz.timezone('Africa/Lagos')) + timedelta(days=30)).isoformat()
}
um.add_user(user_data)
users = um.get_users()
print(users.get("12345"))
um.update_user("12345", {"capital": 3000.0, "active_sessions": ["Tokyo"], "preferred_pairs": ["USD/JPY"]})
um.delete_user("12345")