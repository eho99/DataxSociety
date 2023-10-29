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

class DataState(State):
    """Return what databases the user has access to, and let them select one to use for crowdsourcing

    Returns: List of databases (represented as strings), that a user has access to
    """
    error_message: str = ""
    redirect_to: str = ""

    def on_submit(self, form_data) -> rx.event.EventSpec:
        """Handle login form on_submit.

        Args:
            form_data: A dict of form fields and values.
        """
        self.error_message = ""
        data_table = form_data["data"]
        labels = form_data["label"]

        # with rx.session() as session: 

        # with rx.session() as session:
        #     self.data = session.exec(session.query(project_data).where(project_data.data == data_table_name)).one_or_none()
        #     if not self.data:
        #         self.error_message = (f"Username {data_table} does not exist. Please make a project with using this table to re-access it.")

        # if labels and not self.data:
        #     self.error_message = "This data has only labels and no data. "
        #     return 
        # if self.data and not labels: 
        #     self.error_message = "This data table has no classification labels."
        
        return DataState.redir()  # type: ignore

            
            



