import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import json
import sys
sys.path.append(".")

from visualisation.pages import (
    data_insights_layout,
    # model_performance_layout,
    embedding_visualisation_layout,
    # dataset_suggester_layout,
)
import visualisation.pages.data_insights as data_insights

# import visualisation.pages.model_performance as model_performance
from visualisation.pages.embedding_visualisation import EmbeddingVisualiser

# import visualisation.pages.dataset_suggester as dataset_suggester


class App:
    def __init__(self, dataframes_dict, root_paths, color_dict=None):
        self.dataframes_dict = dataframes_dict
        self.root_paths = root_paths
        self.color_dict = color_dict
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.LUX],
            suppress_callback_exceptions=True,
        )
        self.init_layout()
        self.init_callbacks()

    def init_layout(self):
        navbar = dbc.NavbarSimple(
            children=[
                dbc.NavItem(
                    dcc.Link(
                        "Data Insights", href="/data_insights", className="nav-link"
                    )
                ),
                # dbc.NavItem(
                #     dcc.Link(
                #         "Model Performance",
                #         href="/model_performance",
                #         className="nav-link",
                #     )
                # ),
                dbc.NavItem(
                    dcc.Link(
                        "Embedding Visualisation",
                        href="/embedding_visualisation",
                        className="nav-link",
                    )
                ),
                # dbc.NavItem(
                #     dcc.Link(
                #         "Dataset Suggestion",
                #         href="/dataset_suggestion",
                #         className="nav-link",
                #     )
                # ),
                dbc.NavItem(
                    dcc.Link("Clustering", href="/clustering", className="nav-link")
                ),
            ],
            brand="RAGAAI",
            brand_href="/",
            color="primary",
            dark=True,
        )

        self.app.layout = html.Div(
            children=[
                dcc.Location(id="url"),
                navbar,
                html.Div(
                    id="page-content",
                    style={"margin": "1%"},
                ),
            ],
        )

    def init_callbacks(self):
        data_insights.register_callbacks(self.app)
        # model_performance.register_callbacks(self.app)
        self.embedding_visualiser = EmbeddingVisualiser(
            app=self.app,
            dataframes_dict=self.dataframes_dict,
            root_paths=self.root_paths,
            color_dict=self.color_dict,
        )
        # dataset_suggester.register_callbacks(self.app)

        @self.app.callback(
            Output("page-content", "children"), [Input("url", "pathname")]
        )
        def display_page(pathname):
            if pathname == "/data_insights":
                return data_insights.layout
            # elif pathname == "/model-performance":
            #     return model_performance.layout
            elif pathname == "/embedding_visualisation":
                return self.embedding_visualiser.get_layout()
            # elif pathname == "/dataset-suggestion":
            #     return dataset_suggester.layout
            else:
                return "This is the home page. Please select a page from the navbar."

        # @self.app.callback(
        #     Output("graph", "figure"),
        #     [
        #         Input("dataset-dropdown", "value"),
        #         Input("dimensions-dropdown", "value"),
        #         Input("n-neighbors-slider", "value"),
        #         Input("min-dist-slider", "value"),
        #         Input("marker-size-slider", "value"),
        #         Input("filter-dropdown", "value"),
        #     ],
        # )
        # def update_graph(
        #     dataset_name,
        #     dimensions,
        #     n_neighbors,
        #     min_dist,
        #     marker_size,
        #     selected_option,
        # ):
        #     return self.embedding_visualiser.update_filtered_graph(
        #         dataset_name,
        #         dimensions,
        #         n_neighbors,
        #         min_dist,
        #         marker_size,
        #         selected_option,
        #     )

    def run(self, debug=True, use_reloader=False, port=8051):
        self.app.run_server(debug=debug, use_reloader=use_reloader, port=port)


if __name__ == "__main__":
    # Replace these with your actual dataframes_dict and root_paths
    with open ('visualisation/config.txt', "r") as f:
    # Reading from file
        cfg = json.loads(f.read())   
    dataframes_dict = {}
    root_paths = {}
    for idx, config in enumerate(cfg):
        with open (config['embeddings_path'], "r") as f:
        # Reading from file
            # data1 = json.loads(f.read())    

            dataframes_dict["sample_dataset"+str(idx)] =  pd.DataFrame(json.loads(f.read()))

            root_paths["sample_dataset"+str(idx)]= config["dataset_path"]
    # print(root_paths, dataframes_dict)

    app = App(dataframes_dict=dataframes_dict, root_paths=root_paths)
    app.run(debug=True, use_reloader=False, port=8091)

