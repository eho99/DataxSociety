import reflex as rx

class AppConfig(rx.Config):
    pass

config = AppConfig(
    app_name="DataxSociety",
    db_url="sqlite:///reflex.db",
    env=rx.Env.DEV,
    tailwind={
        "theme": {
            "extend": {
                "backgroundImage": {
                    'image': "url('/bgGif.gif')"
                },
            },
        },
        "plugins": ["@tailwindcss/forms", "@tailwindcss/typography"]
    }
)