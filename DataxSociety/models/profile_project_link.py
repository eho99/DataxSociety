from typing import List, Optional

from sqlmodel import Field, Relationship

import reflex as rx

class UserProfileProjectLink(rx.Model, table=True):
    user_profile_id: Optional[int] = Field(
        default=None, foreign_key="user_profile.id", primary_key=True
    )
    project_id: Optional[int] = Field(
        default=None, foreign_key="project.id"
    )