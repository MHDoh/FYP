"""User management and authentication for the server."""
from werkzeug.security import generate_password_hash, check_password_hash
import json
from pathlib import Path
from typing import Optional, Dict

class UserManager:
    def __init__(self):
        self.users_file = Path(__file__).parent / 'users.json'
        self.users: Dict[str, dict] = self._load_users()

    def _load_users(self) -> Dict[str, dict]:
        """Load users from JSON file."""
        if not self.users_file.exists():
            return {}
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_users(self):
        """Save users to JSON file."""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)

    def add_user(self, username: str, password: str, role: str = 'user') -> bool:
        """Add a new user."""
        if username in self.users:
            return False
        
        self.users[username] = {
            'password_hash': generate_password_hash(password),
            'role': role
        }
        self._save_users()
        return True

    def verify_user(self, username: str, password: str) -> Optional[dict]:
        """Verify user credentials."""
        user = self.users.get(username)
        if user and check_password_hash(user['password_hash'], password):
            return {'username': username, 'role': user['role']}
        return None

    def get_user_role(self, username: str) -> Optional[str]:
        """Get user's role."""
        user = self.users.get(username)
        return user['role'] if user else None

    def update_password(self, username: str, new_password: str) -> bool:
        """Update user's password."""
        if username not in self.users:
            return False
        
        self.users[username]['password_hash'] = generate_password_hash(new_password)
        self._save_users()
        return True