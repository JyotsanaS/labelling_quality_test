import base64
import io
import dash
import numpy as np
import pandas as pd
import umap
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


class VisualiseApp:
    def __init__(self, dataframes_dict, root_paths):
        self.dataframes_dict = dataframes_dict
        self.root_paths = root_paths
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H1("Embedding Visualization", className="text-center"),
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="dataset-dropdown",
                                options=[
                                    {"label": dataset_name, "value": dataset_name}
                                    for dataset_name in dataframes_dict.keys()
                                ],
                                value=list(dataframes_dict.keys())[0],
                                className="mb-3",
                                style={"color": "black"},
                            ),
                            width=4,
                        ),
                        dbc.Col(
                            dcc.RadioItems(
                                id="dimensions",
                                options=[
                                    {"label": "2D", "value": "2D"},
                                    {"label": "3D", "value": "3D"},
                                ],
                                value="2D",
                                inline=True,
                                labelStyle={"margin-right": "10px"},
                                className="mb-3",
                            ),
                            width=4,
                        ),
                    ],
                    justify="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Loading(
                                dcc.Graph(
                                    id="embedding-graph", style={"height": "100%"}
                                ),
                                type="circle",
                                fullscreen=False,
                            ),
                            width=12,
                            className="h-100",
                        )
                    ]
                ),
                dcc.Store(id="umap-cache"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("UMAP n_neighbors:"),
                                dcc.Slider(
                                    id="n-neighbors-slider",
                                    min=2,
                                    max=50,
                                    value=15,
                                    marks={i: f"{i}" for i in range(2, 51, 4)},
                                    step=1,
                                ),
                            ],
                            width=4,
                            className="mx-4",
                        ),
                        dbc.Col(
                            [
                                html.Label("UMAP min_dist:"),
                                dcc.Slider(
                                    id="min-dist-slider",
                                    min=0.0,
                                    max=1.0,
                                    value=0.1,
                                    marks={i / 10: f"{i/10}" for i in range(0, 11, 1)},
                                    step=0.1,
                                ),
                            ],
                            width=4,
                            className="mx-4",
                        ),
                        dbc.Col(
                            [
                                html.Label("Marker size:"),
                                dcc.Slider(
                                    id="marker-size-slider",
                                    min=1,
                                    max=20,
                                    value=8,
                                    marks={i: f"{i}" for i in range(4, 21, 2)},
                                    step=1,
                                ),
                            ],
                            width=4,
                            className="mx-4",
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(id="selected-data"),
                            width=12,
                            className="mb-4 text-center",
                            style={"max-height": "5vh", "overflow": "auto"},
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Save selected data:", className="mr-2"),
                                dbc.Input(
                                    id="csv-filename",
                                    placeholder="Enter file name...",
                                    className="mr-2",
                                ),
                                dbc.Button(
                                    "Save",
                                    id="save-button",
                                    color="success",
                                    className="mr-2",
                                ),
                                html.Div(id="save-status"),
                            ],
                            width=12,
                            className="d-flex align-items-center justify-content-center",
                        ),
                    ],
                    className="my-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(id="selected-images"),
                            width=12,
                            className="text-center",
                            style={"max-height": "30vh", "overflow": "auto"},
                        )
                    ]
                ),
            ],
            fluid=True,
        )

        self.set_callbacks()

    def get_2d_3d_embeddings(self, embedding_df, n_neighbors=15, min_dist=0.1):
        embedding = np.stack(embedding_df["embedding"].dropna())
        reducer_2d = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist
        )
        reducer_3d = umap.UMAP(
            n_components=3, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist
        )

        embedding_2d = reducer_2d.fit_transform(embedding)
        embedding_3d = reducer_3d.fit_transform(embedding)
        return embedding_2d, embedding_3d

    def set_callbacks(self):
        self.app.callback(
            Output("embedding-graph", "figure"),
            [
                Input("dataset-dropdown", "value"),
                Input("dimensions", "value"),
                Input("n-neighbors-slider", "value"),
                Input("min-dist-slider", "value"),
                Input("marker-size-slider", "value"),
            ],
        )(self.update_graph)

        self.app.callback(
            Output("selected-data", "children"),
            [Input("embedding-graph", "selectedData")],
        )(self.display_selected_data)

        self.app.callback(
            Output("selected-images", "children"),
            [
                Input("embedding-graph", "selectedData"),
                Input("dataset-dropdown", "value"),
            ],
        )(self.display_selected_images)

        self.app.callback(
            Output("save-status", "children"),
            [Input("save-button", "n_clicks")],
            [
                State("csv-filename", "value"),
                State("embedding-graph", "selectedData"),
                State("dataset-dropdown", "value"),
            ],
        )(self.save_selected_data_to_csv)

    def update_graph(
        self, dataset_name, dimensions, n_neighbors, min_dist, marker_size
    ):
        dataset_df = self.dataframes_dict[dataset_name]
        embedding_2d, embedding_3d = self.get_2d_3d_embeddings(
            dataset_df, n_neighbors=n_neighbors, min_dist=min_dist
        )

        # create hovertext with id, pred and filepath
        hover_text = dataset_df[["id", "pred", "filepath"]].values.tolist()
        hover_text = [
            f"ID: {ht[0]}<br>Predicted Class: {ht[1]}<br>Name: {ht[2]}"
            for ht in hover_text
        ]

        if dimensions == "2D":
            fig = px.scatter(
                dataset_df.dropna(),
                x=embedding_2d[:, 0],
                y=embedding_2d[:, 1],
                custom_data=["id"],
                color="pred",
                hover_name="id",
                hover_data={"id": True, "pred": True, "filepath": True},
            )
        else:
            fig = px.scatter_3d(
                dataset_df.dropna(),
                x=embedding_3d[:, 0],
                y=embedding_3d[:, 1],
                z=embedding_3d[:, 2],
                custom_data=["id"],
                color="pred",
                hover_name="id",
                hover_data={"id": True, "pred": True, "filepath": True},
            )

        fig.update_traces(marker=dict(size=marker_size), selector=dict(mode="markers"))
        fig.update_layout(
            dragmode="lasso", clickmode="event+select", hovermode="closest"
        )
        fig.update_traces(hovertext=hover_text)

        return fig

    def save_selected_data_to_csv(
        self, n_clicks, filename, selected_data, dataset_name
    ):
        if n_clicks is None or n_clicks == 0:
            return None

        if selected_data is None:
            return "No data selected. Please select some points before saving."

        dataset_df = self.dataframes_dict[dataset_name]
        selected_ids = [point["customdata"][0] for point in selected_data["points"]]
        selected_filepaths = dataset_df[dataset_df["id"].isin(selected_ids)]["filepath"]

        selected_df = pd.DataFrame({"id": selected_ids, "filepath": selected_filepaths})

        selected_df.to_csv(filename, index=False)
        if filename is not "":
            return f"Selected data saved to {filename}."

    def display_selected_data(self, selected_data):
        if selected_data is None:
            return "No data selected"
        else:
            selected_ids = [point["customdata"] for point in selected_data["points"]]
            return f"Selected IDs: {', '.join(map(str, [x[0] for x in selected_ids]))}"

    def display_selected_images(self, selected_data, dataset_name):
        if selected_data is None:
            return "No images selected"

        dataset_df = self.dataframes_dict[dataset_name]
        selected_ids = [point["customdata"][0] for point in selected_data["points"]]
        image_filepaths = dataset_df[dataset_df["id"].isin(selected_ids)][
            "filepath"
        ].tolist()

        images_div = []

        for img_path in image_filepaths:
            with open(f"{self.root_paths[dataset_name]}/{img_path}", "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode("ascii")
            images_div.append(
                html.Img(
                    src=f"data:image/png;base64,{encoded_image}",
                    style={
                        "max-width": "150px",
                        "max-height": "150px",
                        "padding": "5px",
                    },
                )
            )

        return images_div

    def run(self, debug=False):
        self.app.run_server(debug=debug, use_reloader=False, port=8051)


# Example usage
if __name__ == "__main__":
    # Dummy data
    data1 = {
        "id": [x for x in range(1000)],
        "embedding": [
            [0.1 + x / 10, 0.2 + x / 10, 0.3 + x / 10] for x in range(1000, 2000)
        ],
        "filepath": ["path1"] * 1000,
    }
    data2 = {
        "id": [x for x in range(2000, 3000)],
        "embedding": [
            [0.1 + x / 10, 0.2 + x / 10, 0.3 + x / 10] for x in range(1000, 2000)
        ],
        "filepath": ["path4"] * 1000,
    }

    dataframes_dict = {
        "Dataset 1": pd.DataFrame(data1),
        "Dataset 2": pd.DataFrame(data2),
    }

    visualise_app = VisualiseApp(dataframes_dict)
    visualise_app.run(debug=True)
