import reflex as rx

from .base_state import State
from ..models.user import User


class CreateProjectState(State):

    error_message: str = ""
    redirect_to: str = ""

    def on_submit(self, form_data) -> rx.event.EventSpec:
        
        project_name = form_data["project_name"]
        description = form_data["description"]
        num_indep_vars = form_data["num_indep_vars"]
        num_dep_vars = form_data["num_dep_vars"]
        num_data = form_data["num_data"]
        
        
        return CreateProjectState.redir()  

    def redir(self) -> rx.event.EventSpec: 
        return rx.redirect("/")