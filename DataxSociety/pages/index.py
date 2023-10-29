import reflex as rx 
import os

from ..state.login_state import require_login
from ..components.navbar import navbar

def nn_gif(): 
    return rx.image(
        src="/nn_sample_homepage.gif",  
        width="70%"
    )

# def introduction():
#     return rx.box(
#         rx.heading("Welcome to CrowdControlled!"),
#         rx.
#     )

@require_login
def index() -> rx.Component:
    """Render the index page.

    Returns:
        A reflex component.
    """
    # return rx.container(
    #     navbar(),
    #     nn_gif(),
    #     introduction(),
    #     direction="column", 
    # )
    return rx.fragment(
        navbar(),
        nn_gif(),
        rx.box(
            rx.heading("Welcome to my homepage!"),
        ),
    )