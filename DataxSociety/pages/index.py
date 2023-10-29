import reflex as rx 

from ..state.login_state import require_login
from ..components.navbar import navbar

@require_login
def index() -> rx.Component:
    """Render the index page.

    Returns:
        A reflex component.
    """
    return rx.fragment(
        navbar(),
        rx.box(
            rx.heading("Welcome to my homepage!"),
        ),
    )