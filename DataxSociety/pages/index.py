import reflex as rx 

def index() -> rx.Component:
    """Render the index page.

    Returns:
        A reflex component.
    """
    return rx.fragment(
        rx.vstack(
            rx.heading("Welcome to my homepage!"),
            rx.link("Protected Page", href="/protected"),
        ),
    )