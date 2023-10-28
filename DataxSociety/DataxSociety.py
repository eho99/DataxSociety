"""Welcome to Reflex!."""

from DataxSociety import styles

# Import all the pages.
from DataxSociety.pages import *
from .state.base import State

import reflex as rx

# Create the app and compile it.
app = rx.App(state=State, style=styles.base_style)

app.add_page(login, route="/login")
app.add_page(signup, route="/signup")
app.add_page(index, route="/") # , on_load=State.check_login())

app.compile()
