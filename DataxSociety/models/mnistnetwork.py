from typing import Optional, List
from sqlmodel import Field 

import reflex as rx

class MNISTNetwork(rx.Model, table=True):
    """MNISTNetwork Model"""
    id: Optional[int] = Field(default=None, primary_key=True)
    creator_id: int = Field(default=None, foreign_key="userprofile.id") # Many to One
    num_hidden_layers: int = Field(default=None)
    layer_nodes: List[int] = Field(default=None)
    layer_activations: List[str] = Field(default=None)
    learning_rate: float = Field(default=1e-3)
    loss_func: str = Field(default="CrossEntropyLoss")
    num_epochs: int = Field(default=1e4)
    test_ratio: float = Field(default=0.2)
    network_type: str = Field(default="DENSE")
    dropout: float = Field(default=None)
    accuracy: float = Field(default=None)

    