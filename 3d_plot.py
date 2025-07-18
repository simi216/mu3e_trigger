import keras
import tensorflow as tf
import numpy as np
import torch
import matplotlib.pyplot as plt

def convert_pixel_id_to_nla(pixel_id: np.ndarray, padding_value: int = -1) -> np.ndarray:
    nla = np.full((*pixel_id.shape, 4), padding_value, dtype=np.int32)
    valid_mask = pixel_id != padding_value

    chip_id = pixel_id // 2**16
    station = chip_id // 2**12
    layer = ((chip_id // 2**10) % 4) + 1
    phi = ((chip_id // 2**5) % 2**5) + 1
    z_prime = chip_id % 2**5

    z = np.where(layer == 3, z_prime - 7, np.where(layer == 4, z_prime - 6, z_prime))

    station_mask = (station == 0)
    valid_mask = valid_mask & station_mask

    nla[valid_mask, 0] = station[valid_mask]
    nla[valid_mask, 1] = layer[valid_mask]
    nla[valid_mask, 2] = phi[valid_mask]
    nla[valid_mask, 3] = z[valid_mask]

    return nla

def convert_nla_to_location(nla: np.ndarray, padding_value: int = -1) -> np.ndarray:
    location = np.full((*nla.shape[:-1], 3), padding_value, dtype=np.float64)
    valid_mask = ~(nla[:,:, ] == padding_value).any(axis=-1)
    layer = nla[:, : , 1]
    phi = nla[:, : , 2]
    z = nla[:, : , 3]
    print(np.unique(layer))
    #### Define the paramters of the detector layers
    r_layer_1 = 23.3
    r_layer_2 = 29.8
    r_layer_3 = 73.9
    r_layer_4 = 86.3
    length_layer_1 = 124.7
    length_layer_2 = 124.7
    length_layer_3 = 351.9
    length_layer_4 = 372.6
    nz_layer_1 = 6
    nz_layer_2 = 6
    nz_layer_3 = 17
    nz_layer_4 = 18

    nphi_layer_1 = 8
    nphi_layer_2 = 10
    nphi_layer_3 = 24
    nphi_layer_4 = 28

    #### Calculate the z-coordinate in the detector
    location[layer == 1, 2] = ((z[layer == 1])/ nz_layer_1 - 0.5 ) * length_layer_1
    location[layer == 2, 2] = ((z[layer == 2])/ nz_layer_2 - 0.5 ) * length_layer_2
    location[layer == 3, 2] = ((z[layer == 3])/ nz_layer_3 - 0.5 ) * length_layer_3
    location[layer == 4, 2] = ((z[layer == 4])/ nz_layer_4 - 0.5 ) * length_layer_4

    #### Calculate the x-coordinate in the detector
    location[layer == 1, 0] = r_layer_1 * np.cos((phi[layer == 1]) / nphi_layer_1 * 2 * np.pi)
    location[layer == 2, 0] = r_layer_2 * np.cos((phi[layer == 2]) / nphi_layer_2 * 2 * np.pi)
    location[layer == 3, 0] = r_layer_3 * np.cos((phi[layer == 3]) / nphi_layer_3 * 2 * np.pi)
    location[layer == 4, 0] = r_layer_4 * np.cos((phi[layer == 4]) / nphi_layer_4 * 2 * np.pi)

    #### Calculate the y-coordinate in the detector
    location[layer == 1, 1] = r_layer_1 * np.sin((phi[layer == 1]) / nphi_layer_1 * 2 * np.pi)
    location[layer == 2, 1] = r_layer_2 * np.sin((phi[layer == 2]) / nphi_layer_2 * 2 * np.pi)
    location[layer == 3, 1] = r_layer_3 * np.sin((phi[layer == 3]) / nphi_layer_3 * 2 * np.pi)
    location[layer == 4, 1] = r_layer_4 * np.sin((phi[layer == 4]) / nphi_layer_4 * 2 * np.pi)

    return location


def load_ragged_csv_to_ndarray(file_name: str, delimiter: str = ",", fill_value = -1, max_cols = 256, dtype = int) -> np.ndarray:
    rows = []
    row_lengths = []
    with open(file_name, 'r') as file:
        for line in file:
            # Split the line by the delimiter and strip whitespace
            row = np.array([value.strip() for value in line.strip().split(delimiter) if value != ''], dtype=dtype)
            # Ensure the row has at most max_cols elements
            if len(row) > max_cols:
                continue
            rows.append(row)
            row_lengths.append(len(row))
    # Convert the list of rows to a 2D NumPy array
    ragged_array = np.full((len(rows), max_cols), fill_value, dtype=dtype)
    for i, row in enumerate(rows):
        ragged_array[i, :len(row)] = row
    return ragged_array

DATA_DIR = "mu3e_trigger_data"
SIGNAL_DATA_FILE = f"{DATA_DIR}/run42_sig.csv"
BACKGROUND_DATA_FILE = f"{DATA_DIR}/run42_bg.csv"

signal_data_ids = load_ragged_csv_to_ndarray(SIGNAL_DATA_FILE, delimiter=",", fill_value=-1, max_cols=256, dtype=int)
background_data_ids = load_ragged_csv_to_ndarray(BACKGROUND_DATA_FILE, delimiter=",", fill_value=-1, max_cols=256, dtype=int)

signal_data = convert_pixel_id_to_nla(signal_data_ids)

signal_data_location = convert_nla_to_location(signal_data)

# Plot a 3D scatter plot for one event
event_index = 0  # Index of the event to plot
event_location = signal_data_location[event_index]

# Filter out padding values
valid_mask = ~(event_location == -1).any(axis=-1)
event_location = event_location[valid_mask]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(event_location[:, 0], event_location[:, 1], event_location[:, 2], c='b', marker='o')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title(f'3D Plot of Event {event_index}')

plt.show()