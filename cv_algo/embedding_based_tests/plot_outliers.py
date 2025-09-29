import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_outlier_images(results):
    # Create a figure and specify the number of rows and columns in the grid
    num_rows = 5
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 14))

    # List of image file paths
    image_files = results[: (num_rows * num_cols)]["filepath"].values.tolist()
    scores = results[: (num_rows * num_cols)]["score"].values.tolist()
    categories = results[: (num_rows * num_cols)]["label"].values.tolist()
    i = 0

    # Loop through the image files and display them in subplots
    for file, score, category in zip(image_files, scores, categories):
        if i >= (num_rows * num_cols):
            break  # Break if we have displayed all available subplots
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]

        # Load and display the image
        img = mpimg.imread(file)
        ax.imshow(img)

        i += 1
        # Customize subplot (optional)
        ax.set_title(f"score: {score:.2f} label: {category}")
        ax.axis("off")  # Turn off axis labels

    # Remove any remaining empty subplots
    for i in range(len(image_files), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    # Adjust layout and display the grid of images
    plt.tight_layout()
    plt.show()
