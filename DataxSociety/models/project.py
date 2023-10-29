from typing import Optional, List
from sqlmodel import Field, Relationship
from passlib.context import CryptContext

from .user import User
from .user_profile import UserProfile
from .profile_project_link import UserProfileProjectLink

import reflex as rx

class Project(rx.Model, table=True):
    """User Model"""
    id: Optional[int] = Field(default=None, primary_key=True)
    project_name: str = Field(nullable=False)    
    description: str = Field(nullable=False)
    num_datapoints: Optional[int] = Field(default=None) # if None, then unlimited data points are allowed
    num_indep_vars: int = Field(default=0)
    num_dep_vars: int = Field(default=2)
    
    creator_id: UserProfile = Field(default=None, foreign_key="user_profile.id") # Many to One 
    contributors: Optional[List[User]] = Relationship(back_populates="user_contributions", link_table=UserProfileProjectLink) # Many to Many relationship
    