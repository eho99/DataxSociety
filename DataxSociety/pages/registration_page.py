import reflex as rx
from ..state.registration_state import RegistrationState


def registration_page() -> rx.Component:
    """Render the registration page.

    Returns:
        A reflex component.
    """
    register_form = rx.form(
        rx.box(
            rx.form_label("Username", html_for="username", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.input(id="username", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                class_name="mt-2"
            ),
            is_required=True
        ),
        rx.box(
            rx.form_label("Password", html_for="password", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.password(id="password", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                class_name="mt-2"
            ),
            is_required=True
        ),
        rx.box(
            rx.form_label("Confirm password", html_for="confirm_password", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.password(id="confirm_password", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                class_name="mt-2"
            ),
            is_required=True
        ),
        rx.button("Sign Up", type_="submit", class_name="flex w-full justify-center rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"),
        on_submit=RegistrationState.handle_registration,
        class_name="space-y-6"
    )
    return rx.fragment(
        rx.box(
            rx.box(
                rx.heading("Create an account", size="md", class_name="mt-10 text-center text-2xl font-bold leading-9 tracking-tight text-gray-900"),
                class_name="sm:mx-auto sm:w-full sm:max-w-sm"
            ),
            rx.cond(
                RegistrationState.success,
                rx.vstack(
                    rx.text("Registration successful!"),
                    rx.spinner(),
                ),
                rx.box(
                    rx.cond(  # conditionally show error messages
                        RegistrationState.error_message != "",
                        rx.text(RegistrationState.error_message),
                    ),
                    register_form,
                    class_name="mt-10 sm:mx-auto sm:w-full sm:max-w-sm"
                ),
            ),
            class_name="flex min-h-full flex-col justify-center px-6 py-12 lg:px-8"
        )
    )