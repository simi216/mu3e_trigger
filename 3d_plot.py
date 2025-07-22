import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = "mu3e_trigger_data"
SIGNAL_DATA_FILE = f"{DATA_DIR}/run42_sig_positions.npy"
BACKGROUND_DATA_FILE = f"{DATA_DIR}/run42_bg_positions.npy"

max_barrel_radius = 86.3
max_endcap_distance = 372.6

signal_data = np.load(SIGNAL_DATA_FILE)
background_data = np.load(BACKGROUND_DATA_FILE)

background_data[background_data[:,:,0] != -1, 0] /= max_barrel_radius
background_data[background_data[:,:,0] != -1, 1] /= max_barrel_radius
background_data[background_data[:,:,0] != -1, 2] /= max_endcap_distance

signal_data[signal_data[:,:,0] != -1, 0] /= max_barrel_radius
signal_data[signal_data[:,:,0] != -1, 1] /= max_barrel_radius
signal_data[signal_data[:,:,0] != -1, 2] /= max_endcap_distance


def validate_user_input(user_input, data_shape):
    """Validate the event index against the data shape."""
    if not user_input.isdigit():
        return False
    event_index = int(user_input)
    return 0 <= event_index < data_shape[0]

def main():
    user_input = input("Enter the event index to plot: ")
    while(validate_user_input(user_input, signal_data.shape) is False):
        print(f"Invalid event index. Please enter a number between 0 and {signal_data.shape[0] - 1}.")
        user_input = input("Enter the event index to plot: ")
    
    event_index = int(user_input)
    # Plot a 3D scatter plot for one event
    event_location = background_data[event_index]

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

if __name__ == "__main__":
    main()