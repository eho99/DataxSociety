from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator


import reflex as rx

from ..models.user import User
from .login_state import LOGIN_ROUTE, REGISTER_ROUTE
from .base_state import State

class RegistrationState(State):
    """Handle registration form submission and redirect to login page after registration."""

    success: bool = False
    error_message: str = ""

    async def handle_registration(
        self, form_data
    ) -> AsyncGenerator[rx.event.EventSpec | list[rx.event.EventSpec] | None, None]:
        """Handle registration form on_submit.

        Set error_message appropriately based on validation results.

        Args:
            form_data: A dict of form fields and values.
        """
        with rx.session() as session:
            username = form_data["username"]
            if not username:
                self.error_message = "Username cannot be empty"
                yield rx.set_focus("username")
                return
            existing_user = session.exec(
                User.select.where(User.username == username)
            ).one_or_none()
            if existing_user:
                self.error_message = (
                    f"Username {username} is already registered. Try a different name"
                )
                yield [rx.set_value("username", ""), rx.set_focus("username")]
                return
            password = form_data["password"]
            if not password:
                self.error_message = "Password cannot be empty"
                yield rx.set_focus("password")
                return
            if password != form_data["confirm_password"]:
                self.error_message = "Passwords do not match"
                yield [
                    rx.set_value("confirm_password", ""),
                    rx.set_focus("confirm_password"),
                ]
                return
            # Create the new user and add it to the database.
            new_user = User()  # type: ignore
            new_user.username = username
            new_user.password_hash = User.hash_password(password)
            new_user.enabled = True
            session.add(new_user)
            session.commit()
        # Set success and redirect to login page after a brief delay.
        self.error_message = ""
        self.success = True
        yield
        await asyncio.sleep(0.5)
        yield [rx.redirect(LOGIN_ROUTE), RegistrationState.set_success(False)]