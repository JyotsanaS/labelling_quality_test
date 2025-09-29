import pandas as pd

def get_bbox_coordinates(annotations):
    """
    Extracts xmin, ymin, xmax, ymax from a bbox string.

    Args:
    bbox_str (str): Bounding box string in the format of x_centre, y_centre, width, height.

    Returns:
    tuple: (xmin, ymin, xmax, ymax)
    """
    # Assuming we only need the first bounding box
    bbox = annotations['bbox']
    x_centre, y_centre, w, h = bbox
    
    # Calculate xmin, ymin, xmax, ymax
    xmin = x_centre - w / 2
    ymin = y_centre - h / 2
    xmax = x_centre + w / 2
    ymax = y_centre + h / 2
    
    return xmin, ymin, xmax, ymax