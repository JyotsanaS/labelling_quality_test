import base64
import io
import dash
import math
import numpy as np
import pandas as pd
import umap
import plotly.express as px
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from PIL import Image
from io import BytesIO

from visualisation.pages.embedding_visualisation_layout import (
    get_embedding_visualisation_layout,
    get_hover_card_layout,
)


class EmbeddingVisualiser:
    def __init__(self, app, dataframes_dict, root_paths, color_dict=None):
        self.dataframes_dict = dataframes_dict
        self.root_paths = root_paths
        self.app = app
        self.color_dict = color_dict

        self.set_callbacks()
        self.layout = html.Div(
            [
                get_embedding_visualisation_layout(self.dataframes_dict),
                dcc.Tooltip(id="embedding-graph-tooltip"),
                html.Div(
                    [
                        dbc.Pagination(id="image-pagination", max_value=5),
                        html.Div(id="selected-images"),
                    ],
                    style={"display": "flex", "flex-direction": "column"},
                ),
            ]
        )

    def get_layout(self):
        return self.layout

    def filter_data(self, dataset_name, option):
        dataset_df = self.dataframes_dict[dataset_name]
        if option == "all":
            return dataset_df
        elif option == "true_positive":
            return dataset_df[dataset_df["label"] == dataset_df["pred"]]
        elif option == "false_positive":
            return dataset_df[dataset_df["label"] != dataset_df["pred"]]
        elif option == "true_negative":
            return dataset_df[dataset_df["label"] == dataset_df["pred"]]
        elif option == "false_negative":
            return dataset_df[dataset_df["label"] != dataset_df["pred"]]

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
            Input("run-button", "n_clicks"),
            [
                State("dataset-dropdown", "value"),
                State("dimensions", "value"),
                State("n-neighbors-slider", "value"),
                State("min-dist-slider", "value"),
                State("marker-size-slider", "value"),
                State("filter-dropdown", "value"),
            ],
        )(self.update_graph)

        self.app.callback(
            Output("selected-images", "children"),
            [
                Input("embedding-graph", "selectedData"),
                Input("dataset-dropdown", "value"),
                Input("image-pagination", "active_page"),
            ],
            [State("image-pagination", "max_pages")],
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

        self.app.callback(
            Output("embedding-graph-tooltip", "show"),
            Output("embedding-graph-tooltip", "bbox"),
            Output("embedding-graph-tooltip", "children"),
            Input("embedding-graph", "hoverData"),
            State("dataset-dropdown", "value"),
        )(self.display_hover_image)

        self.app.callback(
            Output("image-pagination", "max_value"),
            Input("embedding-graph", "selectedData"),
        )(self.update_pagination_max_value)

    def update_graph(
        self,
        n_clicks,
        dataset_name,
        dimensions,
        n_neighbors,
        min_dist,
        marker_size,
        selected_option,
    ):
        dataset_df = self.filter_data(dataset_name, selected_option)

        embedding_2d, embedding_3d = self.get_2d_3d_embeddings(
            dataset_df, n_neighbors=n_neighbors, min_dist=min_dist
        )

        # create hovertext with id, pred and filepath
        if dimensions == "2D":
            fig = px.scatter(
                dataset_df.dropna(),
                x=embedding_2d[:, 0],
                y=embedding_2d[:, 1],
                custom_data=["id"],
                color="label",
                color_discrete_map=self.color_dict,
            )

            # fig.update_xaxes(fixedrange=True)
            # fig.update_yaxes(fixedrange=True)
        else:
            fig = px.scatter_3d(
                dataset_df.dropna(),
                x=embedding_3d[:, 0],
                y=embedding_3d[:, 1],
                z=embedding_3d[:, 2],
                custom_data=["id"],
                color="label",
                color_discrete_map=self.color_dict,
            )

            fig.update_scenes(dragmode=False)

        fig.update_traces(marker=dict(size=marker_size), selector=dict(mode="markers"))
        fig.update_layout(
            dragmode="lasso", clickmode="event+select", hovermode="closest"
        )
        # fig.update_traces(hovertext=hover_text)

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
        if filename != "":
            return f"Selected data saved to {filename}."

    def display_selected_images(
        self, selected_data, dataset_name, active_page, max_pages
    ):
        if selected_data is None:
            return "No images selected"

        if active_page is None:
            active_page = 1

        dataset_df = self.dataframes_dict[dataset_name]
        selected_ids = [point["customdata"][0] for point in selected_data["points"]]
        selected_images = dataset_df[dataset_df["id"].isin(selected_ids)]

        images_per_page = 100
        start_idx = (active_page - 1) * images_per_page
        end_idx = start_idx + images_per_page
        page_images = selected_images.iloc[start_idx:end_idx]

        row = []
        rows = []

        for i, (idx, row_data) in enumerate(page_images.iterrows()):
            img_path = row_data["filepath"]

            # with open( img_path, "rb") as img_file:
            #     encoded_image = base64.b64encode(img_file.read()).decode("ascii")

            # Load the image using PIL
            with open(img_path, "rb") as img_file:
                image = Image.open(BytesIO(img_file.read()))

            # Convert the image to PNG format
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode("ascii")

            label_text = f"Label: {row_data['label']}, Prediction: {row_data['pred']}"

            image_container = html.Div(
                [
                    html.Div(
                        label_text,
                        style={
                            "position": "absolute",
                            "background-color": "rgba(255, 255, 255, 0.7)",
                            "padding": "2px",
                        },
                    ),
                    html.Img(
                        src=f"data:image/png;base64,{encoded_image}",
                        style={
                            "max-width": "100%",
                            "max-height": "100%",
                            "padding": "5px",
                        },
                    ),
                ],
                style={"position": "relative"},
            )

            row.append(
                dbc.Col(
                    image_container,
                    width={"size": 4, "order": i % 3},
                )
            )

            if (i + 1) % 3 == 0:
                rows.append(dbc.Row(row))
                row = []

        if row:
            rows.append(dbc.Row(row))

        return html.Div(
            rows,
            style={
                "max-height": "600px",
                "overflow-y": "scroll",
                "padding": "10px",
                "border": "1px solid #ccc",
            },
        )

    def display_hover_image(self, hoverData, dataset_name):
        if hoverData is None:
            return False, no_update, no_update

        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        data_id = pt["customdata"][0]

        dataset_df = self.dataframes_dict[dataset_name]
        df_row = dataset_df[dataset_df["id"] == data_id].iloc[0]
        img_path = df_row["filepath"]

        # with open(img_path, "rb") as img_file:
        #     encoded_image = base64.b64encode(img_file.read()).decode("ascii")

        # Load the image using PIL
        with open(img_path, "rb") as img_file:
            image = Image.open(BytesIO(img_file.read()))

        # Convert the image to PNG format
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("ascii")

        img_src = f"data:image/png;base64,{encoded_image}"
        name = df_row["filepath"]
        fid = df_row["id"]
        label = df_row["label"]
        print(df_row["label"])
        pred = df_row["pred"]
        print(df_row["pred"])

        children = [get_hover_card_layout(img_src, fid, label, pred)]

        return True, bbox, children

    def update_pagination_max_value(self, selected_data):
        if selected_data is None:
            return 1

        dataset_name = list(self.dataframes_dict.keys())[0]
        dataset_df = self.dataframes_dict[dataset_name]
        selected_ids = [point["customdata"][0] for point in selected_data["points"]]
        image_filepaths = dataset_df[dataset_df["id"].isin(selected_ids)][
            "filepath"
        ].tolist()

        images_per_page = 100
        total_pages = math.ceil(len(image_filepaths) / images_per_page)

        return total_pages

    # def run(self, debug=False):
    #     self.app.run_server(debug=debug, use_reloader=False, port=8051)


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

    visualise_app = EmbeddingVisualiser(dataframes_dict, root_paths=dataframes_dict)
    visualise_app.run(debug=True)
