import reflex as rx

from .base_state import State
from ..models.user import User


class CreateProjectState(State):

    error_message: str = ""
    redirect_to: str = ""

    def on_submit(self, form_data) -> rx.event.EventSpec:
        
        # TODO
        
        return CreateProjectState.redir()  

    def redir(self) -> rx.event.EventSpec: 
        return rx.redirect("/")