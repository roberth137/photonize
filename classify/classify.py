import pandas as pd
import helper
#import fitting
import numpy as np
#import get_photons
from histogramming import create_histogram_dataframe
import torch
from models import HistogramCNN

folder = '/Users/roberthollmann/Desktop/resi-flim/ml/event_data/'

input_events = f'{folder}cy3_200ms_fp_event_f.hdf5'
input_photons = f'{folder}cy3_59_index.hdf5'
drift_file = f'{folder}cy3_200ms_drift.txt'
NN_model = 'histogram_model_test.pt'
lt_mapping={0: 260, 1: 360, 2: 405}
#fluorophore_number = 0
offset = 10
diameter = 4.5
int_time = 200
bin_size = 20
num_bins = 120
num_classes = 3

#histograms = create_histogram_dataframe(input_events,
#                                        input_photons,
#                                        drift_file,
#                                        offset,
#                                        diameter,
#                                        int_time,
#                                        bin_size=bin_size)

#histograms_np = histograms.to_numpy()

#print(histograms_np.shape)

#histograms = histograms_np[:, :-1]
#events = histograms_np[:, -1]
# Create the model instance
#model_infer = HistogramCNN(num_bins=num_bins, num_classes=num_classes)

# Load state dict
#model_infer.load_state_dict(torch.load("histogram_model_test.pt", weights_only=True))
#model_infer.eval()  # Set to evaluation mode

# Suppose you have new data, 'X_new' (NumPy array)
#X = torch.tensor(histograms, dtype=torch.float32)

#with torch.no_grad():
#    outputs = model_infer(X)
#    _, preds = torch.max(outputs, dim=1)
#    predicted_class = preds.tolist()

#for event, cls in zip(events, predicted_class):
#    print(event, cls)


def predict_and_update_events(events_file,
                              photons_file,
                              drift_file,
                              offset,
                              diameter,
                              int_time,
                              model_path,
                              mapping=lt_mapping):
    """
    Loads events from an HDF5 file, makes predictions with the provided model,
    and adds the following columns:
      - class_lt: mapped from the model prediction using the given mapping.
      - d_class_avg: difference between class_lt and the event's lifetime.
      - dca_n: the square root of (d_class_avg^2).

    Parameters
    ----------
    events_file : str
        Path to the events HDF5 file (assumes key 'locs').
    model : torch.nn.Module
        The pre-loaded PyTorch model for predicting 0, 1, or 2.
    mapping : dict, optional
        Dictionary mapping prediction labels to lifetime values.
        Default is {0: 260, 1: 360, 2: 405}.

    Returns
    -------
    events_df : pd.DataFrame
        Updated events DataFrame with new columns.
    """
    # Load events DataFrame (adjust key as needed)
    events_df = pd.read_hdf(events_file, key='locs')

    histograms = create_histogram_dataframe(events_file,
                                            photons_file,
                                            drift_file,
                                            offset,
                                            diameter,
                                            int_time,
                                            bin_size=bin_size)

    events_df = pd.read_hdf(events_file, key='locs')

    histograms_np = histograms.to_numpy()
    print(histograms_np.shape)

    X_np = histograms_np[:, :-1]  # features for prediction
    event_ids = histograms_np[:, -1]  # event identifiers (optional usage)

    # Create the model instance
    model_infer = HistogramCNN(num_bins=num_bins, num_classes=3)
    model_infer.load_state_dict(torch.load(model_path, weights_only=True))
    model_infer.eval()  # Set to evaluation mode

    # Convert features to a PyTorch tensor
    X_tensor = torch.tensor(X_np, dtype=torch.float32)

    # Run inference on the new data
    with torch.no_grad():
        outputs = model_infer(X_tensor)
        _, preds = torch.max(outputs, dim=1)
        # Convert predictions to a Python list
        predicted_class = preds.tolist()

    # For debugging: print event IDs with predicted classes
    for eid, cls in zip(event_ids, predicted_class):
        print(f"Event {eid} predicted as class {cls}")

    # Create the 'class_lt' column using the mapping dictionary
    # (Here we use predicted_class list, which contains 0,1,or 2 for each event)
    events_df['class_lt'] = [mapping[p] for p in predicted_class]

    # Calculate d_class_avg = class_lt - lifetime
    events_df['d_class_avg'] = events_df['class_lt'] - events_df['lifetime']

    # Calculate dca_n = sqrt((d_class_avg)^2) (equivalent to absolute value)
    events_df['dca_n'] = np.sqrt(events_df['d_class_avg'] ** 2)

    helper.dataframe_to_picasso(events_df, input_events, 'text_mapping', 'first mapping')

    return events_df


events_mapped = predict_and_update_events(input_events,
                                          input_photons,
                                          drift_file,
                                          offset,
                                          diameter,
                                          int_time,
                                          NN_model,
                                          lt_mapping)

print(events_mapped.head())





# --- Example Usage ---
# Paths to your files:
#events_file = '/path/to/your/events_file.hdf5'  # adjust the path accordingly

# Assume 'model' is already loaded (a PyTorch model that outputs 0, 1, or 2).
# For example:
# model = torch.load('your_model_file.pt', map_location=torch.device('cpu'))
# model.eval()

# Run the prediction and update the events DataFrame
#updated_events = predict_and_update_events(events_file, model)

# Now save the updated events DataFrame using the provided function.
#output_filename = 'predicted_events.hdf5'
# You can specify an extension (e.g., '_pred') and optionally pass additional YAML info.
#helper.dataframe_to_picasso(updated_events, output_filename, extension='_pred', yaml_dump="# Predictions added")