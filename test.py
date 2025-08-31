import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Set up the 3D figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('Rotary Position Embedding (RoPE) - 3D Rotation Planes',
             fontsize=16, fontweight='bold')

# Parameters
d_head = 64
base = 10000
max_position = 200
num_pairs = 8

# Select dimension pairs
dimension_indices = np.linspace(0, d_head - 2, num_pairs, dtype=int)
dimension_pairs = [(i, i + 1) for i in dimension_indices]

# Calculate frequencies
frequencies = [1.0 / (base ** (2 * i / d_head)) for i in dimension_indices]

# Colors for each dimension pair
colors = cm.plasma(np.linspace(0, 1, num_pairs))

# Initialize trajectories and planes
trajectories = []
points = []
planes = []

# Set up each dimension pair in its own plane
for i, (dim_pair, freq, color) in enumerate(zip(dimension_pairs, frequencies, colors)):
    z_offset = i * 2.0  # More spacing between planes

    # Create a semi-transparent plane
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 10), np.linspace(-1.5, 1.5, 10))
    zz = np.full_like(xx, z_offset)
    plane = ax.plot_surface(xx, yy, zz, alpha=0.1, color=color)
    planes.append(plane)

    # Add text label for the dimension pair
    ax.text(1.8, 0, z_offset, f'Dims {dim_pair}\nFreq: {freq:.6f}',
            color=color, fontsize=8)

    # Initialize trajectory
    x_data, y_data, z_data = [], [], []
    trajectory, = ax.plot(x_data, y_data, z_data, color=color, linewidth=2)
    trajectories.append(trajectory)

    # Initialize current position point
    point, = ax.plot([], [], [], 'o', color=color, markersize=8)
    points.append(point)

# Set plot properties
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, (num_pairs - 1) * 2 + 1)
ax.set_xlabel('X value')
ax.set_ylabel('Y value')
ax.set_zlabel('Dimension Pair Plane')
ax.view_init(elev=20, azim=45)

# Position indicator
position_text = ax.text2D(0.05, 0.95, "Position: 0", transform=ax.transAxes)

# Store history
history = [[] for _ in range(num_pairs)]


# Animation function
def update(pos):
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        z_offset = i * 2.0

        # Calculate the rotated vector
        angle = pos * freq
        x = np.cos(angle)
        y = np.sin(angle)
        z = z_offset

        # Update history
        history[i].append((x, y, z))

        # Keep only the last 30 positions
        if len(history[i]) > 30:
            history[i].pop(0)

        # Update trajectory
        if len(history[i]) > 1:
            x_vals, y_vals, z_vals = zip(*history[i])
            trajectories[i].set_data(x_vals, y_vals)
            trajectories[i].set_3d_properties(z_vals)

        # Update current position
        points[i].set_data([x], [y])
        points[i].set_3d_properties([z])

    # Update position text
    position_text.set_text(f"Position: {pos}")

    # Rotate the view slightly for a dynamic effect
    ax.view_init(elev=20, azim=45 + pos / 10)

    return trajectories + points + [position_text]


# Create animation
ani = FuncAnimation(fig, update, frames=range(0, max_position, 2),
                    interval=50, blit=True)

plt.tight_layout()
plt.show()