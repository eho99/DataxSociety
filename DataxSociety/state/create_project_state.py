import reflex as rx

from .base_state import State
from ..models.user import User


class CreateProjectState(State):

    error_message: str = ""
    redirect_to: str = ""

    def on_submit(self, form_data) -> rx.event.EventSpec:
        """Handle login form on_submit.

        Args:
            form_data: A dict of form fields and values.
        """
        self.error_message = ""
        username = form_data["username"]
        password = form_data["password"]
        
        return CreateProjectState.redir()  

    def redir(self) -> rx.event.EventSpec: 
        return rx.redirect("/")