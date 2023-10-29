import reflex as rx

from ..state.fetch_data_state import Data_Page_State
from ..state.login_state import require_login
from ..state.base_state import State
from ..components.navbar import navbar
from ..models.project import Project

class SelectState(State):
    option: str = ""

def get_projects():
    with rx.session() as session:
        projects = session.exec(
            session.query(Project).with_entities(Project.project_name)
        ).one_or_none()
    return projects

proj_list = get_projects()
if not proj_list:
    proj_list = []

def create_model_page() -> rx.Component:
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
            rx.form_label("Model Type", html_for="model_type", class_name="block text-sm font-medium leading-6 text-gray-900"),
            rx.box(
                rx.select(
                    ["LSTM", "Dense"],
                    placeholder="Select a model",
                    on_change=SelectState.set_option,
                    id = "model_type"
                ),
                class_name="mt-2",
            ), 
            is_required=True
        ),
        rx.cond(
            SelectState.option == "LSTM", 
            rx.box(
                rx.form_control(
                    rx.form_label("Hidden Size", html_for="hidden_size", class_name="block text-sm font-medium leading-6 text-gray-900"),
                    rx.box(
                        rx.input(id="hidden_size", type="number", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                        class_name="mt-2"
                    ),
                    is_required=True
                ),
                rx.form_control(
                    rx.form_label("Number of layers", html_for="num_layers", class_name="block text-sm font-medium leading-6 text-gray-900"),
                    rx.box(
                        rx.input(id="num_layers", type="number", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                        class_name="mt-2"
                    ),
                    is_required=True
                ),
                rx.form_control(
                    rx.form_label("Dropout", html_for="dropout", class_name="block text-sm font-medium leading-6 text-gray-900"),
                    rx.box(
                        rx.input(id="dropout", type="number", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                        class_name="mt-2"
                    ),
                    is_required=True
                ),
                rx.form_control(
                    rx.form_label("Epochs", html_for="epochs", class_name="block text-sm font-medium leading-6 text-gray-900"),
                    rx.box(
                        rx.input(id="epochs", type="number", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                        class_name="mt-2"
                    ),
                    is_required=True
                ),
                rx.form_control(
                    rx.form_label("Learning rate", html_for="learning_rate", class_name="block text-sm font-medium leading-6 text-gray-900"),
                    rx.box(
                        rx.input(id="num_layers", type="learning_rate", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                        class_name="mt-2"
                    ),
                    is_required=True
                ),
                rx.form_control(
                    rx.form_label("Loss function", html_for="loss_function", class_name="block text-sm font-medium leading-6 text-gray-900"),
                    rx.box(
                        rx.select(
                            ["L1Loss", "MSELoss", "NLLLoss", "CrossEntropyLoss"],
                            placeholder="Select a loss function",
                            on_change=SelectState.set_option,
                            id = "model_type"
                        ),
                        class_name="mt-2"
                    ),
                    is_required=True
                ),
            )
        ),
        rx.cond(
            SelectState.option == "Dense",
            rx.box(
                rx.form_control(
                    rx.form_label("Hidden Sizes", html_for="hidden_sizes", class_name="block text-sm font-medium leading-6 text-gray-900"),
                    rx.box(
                        rx.input(id="hidden_sizes", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                        class_name="mt-2"
                    ),
                    is_required=True
                ),
                rx.form_control(
                    rx.form_label("Activation Functions", html_for="activations", class_name="block text-sm font-medium leading-6 text-gray-900"),
                    rx.box(
                        rx.input(id="activations", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                        class_name="mt-2"
                    ),
                    is_required=True
                ),
                rx.form_control(
                    rx.form_label("Epochs", html_for="epochs", class_name="block text-sm font-medium leading-6 text-gray-900"),
                    rx.box(
                        rx.input(id="epochs", type="number", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                        class_name="mt-2"
                    ),
                    is_required=True
                ),
                rx.form_control(
                    rx.form_label("Learning rate", html_for="learning_rate", class_name="block text-sm font-medium leading-6 text-gray-900"),
                    rx.box(
                        rx.input(id="num_layers", type="learning_rate", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                        class_name="mt-2"
                    ),
                    is_required=True
                ),
                rx.form_control(
                    rx.form_label("Loss function", html_for="loss_function", class_name="block text-sm font-medium leading-6 text-gray-900"),
                    rx.box(
                        rx.select(
                            ["L1Loss", "MSELoss", "NLLLoss", "CrossEntropyLoss"],
                            placeholder="Select a loss function",
                            on_change=SelectState.set_option,
                            id = "model_type"
                        ),
                        class_name="mt-2"
                    ),
                    is_required=True
                ),
                rx.form_control(
                    rx.form_label("Test Ratio", html_for="test_ratio", class_name="block text-sm font-medium leading-6 text-gray-900"),
                    rx.box(
                        rx.input(id="test_ratio", type="number", class_name="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"),
                        class_name="mt-2"
                    ),
                    is_required=True
                ),
            ),
        ),
        rx.button("Create Project", type_="submit", class_name="flex w-full justify-center rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"),
        # on_submit=,
        class_name="space-y-6"
    )

    return rx.cond(
        Data_Page_State.is_hydrated,
        rx.box(
            navbar(),
            rx.box(
                rx.heading("Train a new model"),
                form,  
                class_name="flex min-h-full flex-col justify-center px-6 py-12 lg:px-8"
            ),  
        )
    )
    
    