import numpy as np
from get_photons import crop_cuboid   # wherever your crop_cuboid lives

def extract_spot_histogram(pick_photons, event, box_side_length, t0, t1):
    """
    Crop out a (box_side_length×box_side_length) region around (event.x,event.y)
    between times t0 and t1, and return a 2D histogram of photon counts per pixel.

    Parameters
    ----------
    pick_photons : your photon‐list or DataFrame
    event        : a record/Series with float attributes .x and .y
    box_side_length : int, e.g. 5
    t0, t1       : time bounds (start_ms, end_ms)

    Returns
    -------
    H : (box_side_length, box_side_length) numpy array of counts
    """
    half = box_side_length / 2.0

    # 1) Compute integer pixel bounds
    x0 = int(round(event.x - half))
    x1 = x0 + box_side_length
    y0 = int(round(event.y - half))
    y1 = y0 + box_side_length

    # 2) Crop the photon‐list/DataFrame to that cuboid
    spot_df = crop_cuboid(
        pick_photons,
        x0, x1,
        y0, y1,
        t0, t1
    )

    # 3) Build a 2D histogram of (x,y) into exactly box_side_length bins each
    #    bin edges aligned on integer pixel boundaries:
    xedges = np.arange(x0, x1 + 1)
    yedges = np.arange(y0, y1 + 1)

    H, _, _ = np.histogram2d(
        spot_df.x.values,
        spot_df.y.values,
        bins=[xedges, yedges]
    )

    # H[i,j] is the count in pixel (xedges[i] ≤ x < xedges[i+1], same for y)
    return H.astype(np.float32), xedges[0].astype(np.float32), yedges[0].astype(np.float32)