import reflex as rx 

def index() -> rx.Component:
    """Render the index page.

    Returns:
        A reflex component.
    """
    return rx.fragment(
        rx.color_mode_button(rx.color_mode_icon(), float="right"),
        rx.vstack(
            rx.heading("Welcome to my homepage!", font_size="2em"),
            rx.link("Protected Page", href="/protected"),
            spacing="1.5em",
            padding_top="10%",
        ),
    )