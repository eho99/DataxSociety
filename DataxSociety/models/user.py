from typing import Optional
from sqlmodel import Field, Relationship
from passlib.context import CryptContext

import reflex as rx

from .userprofile import UserProfile

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(rx.Model, table=True):
    """User Model"""
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, nullable=False, index=True)
    password_hash: str = Field(nullable=False)
    enabled: bool = False

    userprofile_id: Optional[int] = Field(default=None, foreign_key="userprofile.id")

    @staticmethod
    def hash_password(secret: str) -> str:
        """Hash the secret using bcrypt.

        Args:
            secret: The password to hash.

        Returns:
            The hashed password.
        """
        return pwd_context.hash(secret)

    def verify(self, secret: str) -> bool:
        """Validate the user's password.

        Args:
            secret: The password to check.

        Returns:
            True if the hashed secret matches this user's password_hash.
        """
        return pwd_context.verify(
            secret,
            self.password_hash,
        )
    