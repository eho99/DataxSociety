"""Index Page: Serves as the landing page/initial page for users who are not signed up."""

import reflex as rx
from ..components.header import header

def index():
    return header()