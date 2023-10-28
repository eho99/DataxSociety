import reflex as rx

def header():
    return rx.box(
        rx.hstack(
            rx.image(src="favicon.ico"),
            rx.heading("My App"),
            rx.link("Home", href="/"),
            rx.link("Login", href="/login"),
            rx.link("Sign Up", href="/signup")
        ),
        position="fixed",
        width="100%",
        top="0px",
        z_index="5",
    )