from typing import Optional, List
from sqlmodel import Field, Relationship

from .userprofile import UserProfile
from .profile_project_link import UserProfileProjectLink

import reflex as rx

class Project(rx.Model, table=True):
    """Project Model"""
    id: Optional[int] = Field(default=None, primary_key=True)
    project_name: str = Field(nullable=False, unique=True)    
    description: str = Field(nullable=False)
    num_datapoints: Optional[int] = Field(default=None) # if None, then unlimited data points are allowed
    num_indep_vars: int = Field(default=0)
    num_dep_vars: int = Field(default=2)
    best_model_id: Optional[int] = Field(default=None, foreign_key="mnistnetwork.id") # will need to update as it goes
    
    creator_id: int = Field(default=None, foreign_key="userprofile.id") # Many to One 
    project_contributors: Optional[List[UserProfile]] = Relationship(back_populates="user_contributions", link_model=UserProfileProjectLink) # Many to Many relationship
    