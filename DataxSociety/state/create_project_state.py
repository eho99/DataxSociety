import reflex as rx

from .base_state import State
from ..models.user import User
from ..models.project import *
from ..models.auth_session import AuthSession
import datetime


class CreateProjectState(State):

    error_message: str = ""
    redirect_to: str = ""

    def on_submit(self, form_data) -> rx.event.EventSpec:
        
        project_name = form_data["project_name"]
        description = form_data["description"]
        num_indep_vars = form_data["num_indep_vars"]
        num_dep_vars = form_data["num_dep_vars"]
        num_data = form_data["num_data"]

        creator = form_data["username"]

        with rx.session() as session:
            result = session.exec(
                session.query(User).where(User.username == creator)
            ).first()
            print(result)
            creator_id = result[0].id
        contributors = []


        new_proj = Project()
        new_proj.project_name = project_name
        new_proj.description = description
        new_proj.num_datapoints = num_data
        new_proj.num_indep_vars = num_indep_vars
        new_proj.num_dep_vars = num_dep_vars
        new_proj.creator_id = creator_id
        new_proj.project_contributors = contributors

        with rx.session() as session:
            session.add(Project(project_name=new_proj.project_name, description=new_proj.description,
                                num_datapoints=new_proj.num_datapoints, num_indep_vars=new_proj.num_indep_vars, 
                                num_dep_vars=new_proj.num_dep_vars, best_model_id=new_proj.best_model_id, creator_id=new_proj.creator_id,
                                project_contributors=new_proj.project_contributors))
            session.commit()
        
        return CreateProjectState.redir()  

    def redir(self) -> rx.event.EventSpec: 
        return rx.redirect("/")