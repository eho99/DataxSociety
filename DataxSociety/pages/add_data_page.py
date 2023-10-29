import reflex as rx

# from ..state.fetch_data_state import DataState
from ..state.login_state import require_login
from ..state.base_state import State
from ..components.navbar import navbar
from ..models.project import Project
from ..models.user import User
from ..state.add_data_state import AddDataState

# Gets a list of projects
def get_projects():
    with rx.session() as session:
        projects = session.exec(
            session.query(Project.project_name)
        ).all()
    return projects

# Gets a list of Users
def get_users():
    with rx.session() as session:
        users = session.exec(
            session.query(User).with_entities(User.username)
        ).one_or_none()
    return users


users = get_users()
if not users:
    users = []
else:
    users = list(users)

proj_list = get_projects()
if not proj_list:
    proj_list = []
else:
    proj_list = [proj[0] for proj in proj_list]

def add_data_page() -> rx.Component:

    

    form = rx.form(
        rx.form_control(
            rx.form_label("Project Name", html_for="project_name", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.select(
                    proj_list,
                    placeholder="Select a Project",
                    id = "project_name"
                ),
                class_name="mt-2",
            ),
            is_required=True
        ),
        rx.form_control(
            rx.form_label("Data", html_for="input_data", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.input(id="input_data", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                class_name="mt-2"
            ), 
            is_required=True
        ),
        rx.form_control(
            rx.form_label("Output Label", html_for="output_label", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.input(id="output_label", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                class_name="mt-2"
            ), 
            is_required=True
        ),
        rx.form_control(
            rx.form_label("User", html_for="username", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.select(
                    users,
                    placeholder="Select your username",
                    id="username"
                ),
                class_name="mt-2",
            ),
            is_required=True
        ),
        rx.button("Add Data", type_="submit", class_name="flex w-full justify-center rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"),
        on_submit=AddDataState.on_submit,
        class_name="space-y-6"
    )

    return rx.cond(
        AddDataState.is_hydrated,
        rx.box(
            navbar(),
            rx.box(
                rx.heading("Add Data to a Project"),
                form,  
                class_name="flex min-h-full flex-col justify-center px-6 py-12 lg:px-8"
            ),  
        )
    )
    
    