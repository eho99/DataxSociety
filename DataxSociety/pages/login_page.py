import reflex as rx
from ..state.login_state import LoginState
from ..state.login_state import LOGIN_ROUTE, REGISTER_ROUTE


def login_page() -> rx.Component:
    """Render the login page.

    Returns:
        A reflex component.
    """
    login_form = rx.form(
        rx.form_control(
            rx.form_label("Username", html_for="username", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.input(id="username", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                class_name="mt-2"
            ),
            is_required=True
        ),
        rx.form_control(
            rx.form_label("Password", html_for="password", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.password(id="password", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                class_name="mt-2"
            ),
            is_required=True
        ),
        rx.button("Sign in", type_="submit", class_name="flex w-full justify-center rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"),
        on_submit=LoginState.on_submit,
        class_name="space-y-6"
    )

    return rx.fragment(
        rx.box(
            rx.box(
                rx.heading("Sign in to your account", size="md", class_name="mt-10 text-center text-2xl font-bold leading-9 tracking-tight text-gray-900"),
                class_name="sm:mx-auto sm:w-full sm:max-w-sm"
            ),
            rx.cond(
                LoginState.is_hydrated,  # type: ignore
                rx.box(
                    rx.cond(  # conditionally show error messages
                        LoginState.error_message != "",
                        rx.text(LoginState.error_message),
                    ),
                    login_form,
                    rx.text("Don't have an account? ", 
                        rx.link("Sign up", href=REGISTER_ROUTE, class_name="font-semibold leading-6 text-indigo-600 hover:text-indigo-500"),
                        class_name="mt-10 text-center text-sm text-gray-500"
                    ),        
                    class_name="mt-10 sm:mx-auto sm:w-full sm:max-w-sm"
                ),
            ),
            class_name="flex min-h-full flex-col justify-center px-6 py-12 lg:px-8"
        )
    )