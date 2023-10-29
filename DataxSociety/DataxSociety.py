"""Main app module to demo local authentication."""
import reflex as rx

from .state.base_state import State
from .state.login_state import require_login, LOGIN_ROUTE, REGISTER_ROUTE
from .pages.registration_page import registration_page as registration_page
from .pages.login_page import login_page
from .pages.create_project_page import create_project_page
from .pages.create_model_page import create_model_page

from .pages.index import index

app = rx.App()

app.add_page(index)
app.add_page(login_page, route=LOGIN_ROUTE)
app.add_page(registration_page, route=REGISTER_ROUTE)
app.add_page(create_project_page, route="/create_project")
app.add_page(create_model_page, route="/train_model")

app.compile()