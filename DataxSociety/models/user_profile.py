from typing import List, Optional
from sqlmodel import Field, Relationship
from .project import Project
from .user import User
from .profile_project_link import UserProfileProjectLink

import reflex as rx

class UserProfile(rx.Model, table=True):
    """User Model"""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: Optional[str] = None
    email: Optional[str] = None

    user_id: Optional[int] = Field(default=None, foreign_key="user.id") # One to One relationship with Users
    user_contributions: Optional[List[Project]] = Relationship(back_populates="project_contributors", link_model=UserProfileProjectLink) # Many to Many relationship


    