"""Main app module to demo local authentication."""
import reflex as rx

from .state.base_state import State
from .state.login_state import require_login, LOGIN_ROUTE, REGISTER_ROUTE
from .pages.registration_page import registration_page as registration_page
from .pages.login_page import login_page

from .pages.index import index


@require_login
def protected() -> rx.Component:
    """Render a protected page.

    The `require_login` decorator will redirect to the login page if the user is
    not authenticated.

    Returns:
        A reflex component.
    """
    return rx.vstack(
        rx.heading(
            "Protected Page for ", State.authenticated_user.username, font_size="2em"
        ),
        rx.link("Home", href="/"),
        rx.link("Logout", href="/", on_click=State.do_logout),
    )

app = rx.App()

app.add_page(index)
app.add_page(protected)
app.add_page(login_page, route=LOGIN_ROUTE)
app.add_page(registration_page, route=REGISTER_ROUTE)

app.compile()