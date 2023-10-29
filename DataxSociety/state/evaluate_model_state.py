"""
State to fetch access to model and data a user has used from a database
"""

from sqlmodel import select

import reflex as rx

from ..models.auth_session import AuthSession
from ..models.user import User
from .login_state import LOGIN_ROUTE, REGISTER_ROUTE
from .base_state import State

class Eval_State(State):
    """Return the accuracy of the model on the given dataset query

    Returns: int representing the accuracy of the model
        """

    def eval_model(self, Dataset, Model):
        
        with rx.session as session():


