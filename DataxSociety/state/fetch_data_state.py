"""
Top-level State for the Projects Page

Databases are stored in a SQL backend in the State class so all substates
can access it to see who has acces to what databases used for event handlers and
computed vars.
"""

from sqlmodel import select

import reflex as rx

from ..models.auth_session import AuthSession
from ..models.user import User
from .login_state import LOGIN_ROUTE, REGISTER_ROUTE
from .base_state import State

class Data_Page_State(State):
    """Return what databases the user has access to, and let them select one to use for crowdsourcing

    Returns: List of databases (represented as strings), that a user has access to
        """

    def fetch_databases(self):
        self.databases = []
        return

