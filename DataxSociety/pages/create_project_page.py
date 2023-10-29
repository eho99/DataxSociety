import reflex as rx

from ..state.login_state import LoginState
from ..state.login_state import require_login
from ..components.navbar import navbar

@require_login
def create_project_page() -> rx.Component:
    return rx.box(
        navbar(),
        rx.heading("Create a new project"),
    )