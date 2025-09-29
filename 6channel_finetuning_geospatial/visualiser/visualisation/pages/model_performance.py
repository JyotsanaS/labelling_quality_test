# from app import app
from dash.dependencies import Input, Output, State

# Import any other required libraries


def register_callbacks(app):
    # Add callback functions here

    # Example callback
    @app.callback(
        Output("output_component", "children"),
        [Input("input_component", "value")],
        allow_duplicates=True,
        prevent_initial_call="initial_duplicate",
    )
    def update_output(input_value):
        return f"You have entered: {input_value}"
