from typing import Optional, List

from sqlmodel import Field

import reflex as rx

class MNISTData(rx.Model, table=True):
    """MNIST Data Model"""
    id: Optional[int] = Field(default=None, primary_key=True)
    creator_id: int = Field(default=None, foreign_key="userprofile.id") # Many to One
    pixel_vals: List[float] = Field(default=None) # 1D array of brightness values for mnist
    label: int = Field(default=None)

    def add_datapoint(self): # EXPECTS pixel_map as a numpy array
        with rx.session as session:
            session.add(MNISTData(id=self.id, creator_id=self.creator_id, pixel_vals=self.pixel_vals, label=self.label))
            session.commit()