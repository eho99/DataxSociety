import reflex as rx

from .base_state import State
from ..models.user import User
from ..models.project import *
from ..models.auth_session import AuthSession
from ..models.mnistdata import MNISTData
import datetime


class AddDataState(State):

    error_message: str = ""
    redirect_to: str = ""

    def on_submit(self, form_data) -> rx.event.EventSpec:
        
        project_name = form_data["project_name"]
        data = form_data["input_data"]
        label = form_data["output_label"]

        creator = form_data["username"]

        with rx.session() as session:
            result = session.exec(
                session.query(User).where(User.username == creator)
            ).first()
            creator_id = result[0].id
        contributors = []

        new_data = MNISTData()
        new_data.pixel_vals = data
        new_data.label = label
        new_data.creator_id = creator_id


        with rx.session() as session:
            session.add(MNISTData(creator_id=new_data.creator_id,
                                  pixel_vals=new_data.pixel_vals,
                                  label=new_data.label))
            session.commit()
        
        return AddDataState.redir()  

    def redir(self) -> rx.event.EventSpec: 
        return rx.redirect("/")