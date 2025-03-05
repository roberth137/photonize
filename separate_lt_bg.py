import numpy as np
import pandas as pd
import helper

filename = 'local/4colors_2/picks_event.hdf5'
separation_points = [(271, 0.487, 299, 0.590),
                     (311, 0.462, 348, 0.573),
                     (370, 0.275, 399, 0.324)]

def add_line_metrics(
    df: pd.DataFrame,
    lines: list
) -> pd.DataFrame:
    """
    Given a DataFrame with columns for lifetime & brightness,
    and a list of lines (each line defined by two points),
    this function adds new columns metric1, metric2, ...,
    where each metric indicates how far above/below the line
    an event is in lifetime/brightness space.

    For line i defined by points (l1, b1) and (l2, b2):
      - slope m = (b2 - b1)/(l2 - l1)      (unless l2 == l1)
      - intercept c = b1 - m*l1
      - metric = brightness - (m*lifetime + c)

    Parameters
    ----------
    df : pd.DataFrame
        Must have lifetime_col and brightness_col for each event.
    lines : list
        A list of lines. Each line is defined as a tuple:
        (l1, b1, l2, b2), specifying two points in (lifetime, brightness) space.
    lifetime_col : str, default="lifetime"
        Name of the column for lifetime data.
    brightness_col : str, default="bg"
        Name of the column for brightness data.

    Returns
    -------
    df_out : pd.DataFrame
        A copy of the original DataFrame with extra columns:
        metric1, metric2, ..., one per input line.
        Positive metric => event is above the line;
        Negative metric => below the line;
        Zero => on the line.
    """

    df_out = df.copy()

    # Convert lifetime, brightness into numpy arrays for faster math.
    lifetime_vals = df_out['lifetime_10ps'].to_numpy()
    brightness_vals = df_out['brightness_norm'].to_numpy()

    # Loop over each line, add a column "metric1", "metric2", ...
    for i, (l1, b1, l2, b2) in enumerate(lines, start=1):
        line_label = f"metric{i}"

        # Handle vertical line edge case
        if np.isclose(l1, l2):
            raise ValueError(
                f"Line {i} is vertical or nearly vertical (l1={l1}, l2={l2}). "
                "Cannot compute slope/intercept in the usual way."
            )

        # Compute slope (m) and intercept (c)
        m = (b2 - b1) / (l2 - l1)
        c = b1 - m * l1

        # Compute metric: b - (m*l + c)
        metric_vals = brightness_vals - (m * lifetime_vals + c)
        df_out[line_label] = metric_vals

    return df_out

if __name__ == "__main__":
    events = pd.read_hdf(filename, key='locs')
    events = add_line_metrics(events, separation_points)
    helper.dataframe_to_picasso(events, filename, '_metrics')