import reflex as rx
from ..state.login_state import LoginState
from ..state.login_state import LOGIN_ROUTE, REGISTER_ROUTE


def login_page() -> rx.Component:
    """Render the login page.

    Returns:
        A reflex component.
    """
    login_form = rx.form(
        rx.input(placeholder="username", id="username"),
        rx.password(placeholder="password", id="password"),
        rx.button("Login", type_="submit"),
        width="80vw",
        on_submit=LoginState.on_submit,
    )

    return rx.fragment(
        rx.cond(
            LoginState.is_hydrated,  # type: ignore
            rx.vstack(
                rx.cond(  # conditionally show error messages
                    LoginState.error_message != "",
                    rx.text(LoginState.error_message),
                ),
                login_form,
                rx.link("Register", href=REGISTER_ROUTE),
                padding_top="10vh",
            ),
        )
    )