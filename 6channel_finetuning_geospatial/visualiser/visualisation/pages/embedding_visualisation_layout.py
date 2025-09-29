from dash import dcc, html
import dash_bootstrap_components as dbc


def get_hover_card_layout(img_src, fid, label, pred):
    hover_card_layout = html.Div(
        [
            html.Img(src=img_src, style={"width": "100%"}),
            html.Div(
                [
                    html.P(
                        f"ID: {fid}",
                        style={
                            "color": "darkblue",
                            "font-size": "11px",
                            "font-weight": "bold",
                            "margin": "5px 0",
                        },
                    ),
                    html.P(
                        f"Label: {label}",
                        style={
                            "color": "darkblue",
                            "font-weight": "bold",
                            "font-size": "11px",
                            "margin": "5px 0",
                        },
                    ),
                    html.P(
                        f"Pred: {pred}",
                        style={
                            "color": "darkblue",
                            "font-weight": "bold",
                            "font-size": "11px",
                            "margin": "5px 0",
                        },
                    ),
                ],
                style={"margin-top": "5px"},
            ),
        ],
        style={
            "width": "200px",
            "white-space": "normal",
            "background-color": "white",
            "border": "1px solid lightgrey",
            "border-radius": "5px",
            "padding": "10px",
            "box-shadow": "2px 2px 5px rgba(0, 0, 0, 0.1)",
        },
    )
    return hover_card_layout


def get_embedding_visualisation_layout(dataframes_dict):
    page_layout = dbc.Container(
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
                        dcc.Dropdown(
                            id="filter-dropdown",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "True Positive", "value": "true_positive"},
                                {"label": "False Positive", "value": "false_positive"},
                                {"label": "True Negative", "value": "true_negative"},
                                {"label": "False Negative", "value": "false_negative"},
                            ],
                            value="all",
                            clearable=False,
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
                            dcc.Graph(id="embedding-graph", style={"height": "100%"}),
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
                        width=3,
                        className="mx-3",
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
                        width=3,
                        className="mx-3",
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
                        width=3,
                        className="mx-3",
                    ),
                    dbc.Col(
                        [
                            # html.Label("Run:"),
                            dbc.Button(
                                "Update",
                                id="run-button",
                                color="primary",
                                className="mt-1",
                            ),
                        ],
                        width=1,
                        className="mx-2",
                    ),
                ],
                className="mb-3",
            ),
            # dbc.Row(
            #     [
            #         dbc.Col(
            #             html.Div(id="selected-data"),
            #             width=12,
            #             className="mb-4 text-center",
            #             style={"max-height": "5vh", "overflow": "auto"},
            #         )
            #     ]
            # ),
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

    return page_layout
