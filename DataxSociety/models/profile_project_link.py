from typing import List, Optional

from sqlmodel import Field, Relationship

import reflex as rx

class UserProfileProjectLink(rx.Model, table=True):
    userprofile_id: Optional[int] = Field(
        default=None, foreign_key="userprofile.id", primary_key=True
    )
    project_id: Optional[int] = Field(
        default=None, foreign_key="project.id", primary_key=True
    )