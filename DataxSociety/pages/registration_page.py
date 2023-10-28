import reflex as rx
from ..state.registration_state import RegistrationState


def registration_page() -> rx.Component:
    """Render the registration page.

    Returns:
        A reflex component.
    """
    register_form = rx.form(
        rx.input(placeholder="username", id="username"),
        rx.password(placeholder="password", id="password"),
        rx.password(placeholder="confirm", id="confirm_password"),
        rx.button("Register", type_="submit"),
        width="80vw",
        on_submit=RegistrationState.handle_registration,
    )
    return rx.fragment(
        rx.cond(
            RegistrationState.success,
            rx.vstack(
                rx.text("Registration successful!"),
                rx.spinner(),
            ),
            rx.vstack(
                rx.cond(  # conditionally show error messages
                    RegistrationState.error_message != "",
                    rx.text(RegistrationState.error_message),
                ),
                register_form,
                padding_top="10vh",
            ),
        )
    )