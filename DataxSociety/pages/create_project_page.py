import reflex as rx

from ..state.create_project_state import CreateProjectState
from ..state.login_state import require_login
from ..components.navbar import navbar

def create_project_page() -> rx.Component:

    form = rx.form(
        rx.form_control(
            rx.form_label("Project Name", html_for="project_name", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.input(id="project_name", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                class_name="mt-2"
            ),
            is_required=True
        ),
        rx.form_control(
            rx.form_label("Description", html_for="description", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.text_area(id="description", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                class_name="mt-2"
            ),
            is_required=True
        ),
        rx.form_control(
            rx.form_label("Number of Independent Variables", html_for="num_indep_vars", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.input(id="num_indep_vars", type="number", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                class_name="mt-2"
            ),
            is_required=True
        ),
        rx.form_control(
            rx.form_label("Number of Dependent Variables", html_for="num_dep_vars", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.input(id="num_dep_vars", type="number", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                class_name="mt-2"
            ),
            is_required=True
        ),
        rx.form_control(
            rx.form_label("Number of data points", html_for="num_data", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.input(id="num_data", type="number", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                class_name="mt-2"
            ),
        ),
        rx.button("Create Project", type_="submit", class_name="flex w-full justify-center rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"),
        on_submit=CreateProjectState.on_submit,
        class_name="space-y-6"
    )

    return rx.cond(
        CreateProjectState.is_hydrated,
        rx.box(
            navbar(),
            rx.heading("Create a new project"),
            form,
        )
    )
    
    