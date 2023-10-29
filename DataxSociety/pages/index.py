import reflex as rx 
import os

from ..state.login_state import require_login
from ..components.navbar import navbar

# def introduction():
#     return rx.box(
#         rx.heading("Welcome to CrowdControlled!"),
#         rx.
#     )

@require_login
def index() -> rx.Component:
    """Render the index page.

    Returns:
        A reflex component.
    """
    # return rx.container(
    #     navbar(),
    #     nn_gif(),
    #     introduction(),
    #     direction="column", 
    # )
    return rx.box(
        navbar(),
        rx.html('''
            <div class="h-screen pb-14 bg-right bg-cover bg-image">
                <!--Main-->
                <div class="container pt-24 md:pt-48 px-6 mx-auto flex flex-wrap flex-col md:flex-row items-center">
                    
                    <!--Left Col-->
                    <div class="flex flex-col w-full xl:w-2/5 justify-center lg:items-start overflow-y-hidden">
                        <h1 class="my-4 text-3xl md:text-5xl text-white font-bold leading-tight text-center md:text-left slide-in-bottom-h1">CrowdControlled - Democratizing Data and Machine Learning</h1>
                        <p class="leading-normal text-white md:text-2xl mb-8 text-center md:text-left slide-in-bottom-subtitle">
                            Harness the power of the collective to break down barriers between data seekers and data providers, enabling both businesses and individuals to embark on data-centric projects and tap into the potential of machine learning. 
                        </p>

                    </div>
                    
                </div>
                

            </div>

        ''')
    )