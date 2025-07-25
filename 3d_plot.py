import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



DATA_DIR = "mu3e_trigger_data"
SIGNAL_DATA_FILE = f"{DATA_DIR}/run42_sig_positions.npy"
BACKGROUND_DATA_FILE = f"{DATA_DIR}/run42_bg_positions.npy"

max_barrel_radius = 86.3
max_endcap_distance = 372.6

signal_data = np.load(SIGNAL_DATA_FILE)
bg_data_positions = np.load(BACKGROUND_DATA_FILE)

def validate_user_input(user_input, data_shape):
    """Validate the event index against the data shape."""
    if not user_input.isdigit():
        return False
    event_index = int(user_input)
    return 0 <= event_index < data_shape[0]

user_input = input("Enter the event index to plot: ")
while(validate_user_input(user_input, signal_data.shape) is False):
    print(f"Invalid event index. Please enter a number between 0 and {signal_data.shape[0] - 1}.")
    user_input = input("Enter the event index to plot: ")

event_index = int(user_input)
# Plot a 3D scatter plot for one event
x, y,z  = bg_data_positions[:event_index, :, 0], bg_data_positions[:event_index, :, 1], bg_data_positions[:event_index, :, 2]
fig = plt.figure(figsize=(10, 9))
ax = plt.axes(projection='3d')
sc = ax.scatter3D(x, y, z, c="blue", marker="o")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D Scatter Plot of Background Data")

# Enable rotation and zoom
def on_move(event):
    if event.button == 1:  # Left mouse button for rotation
        ax.view_init(elev=ax.elev + event.ydata * 0.1, azim=ax.azim + event.xdata * 0.1)
    elif event.button == 3:  # Right mouse button for zoom
        ax.dist += event.ydata * 0.01
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_move)
plt.show()