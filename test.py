import numpy as np
import matplotlib.pyplot as plt

# Parameters
d = 64          # Embedding dimension (per head, for visualization)
max_pos = 50    # Max sequence length
theta_base = 10000

# Compute frequencies (theta_i)
i = np.arange(0, d // 2)
theta = 1.0 / (theta_base ** (2*i / d))

# Position indices
m = np.arange(max_pos)

# Compute angles for each position and dimension: m * theta_i
angles = np.outer(m, theta)  # shape: (max_pos, d//2)

# Expand to full d dimensions (for sin and cos interleaving)
# Each angle corresponds to a 2D rotation: [cos(angle), sin(angle)]
cos_pos = np.cos(angles)  # shape: (max_pos, d//2)
sin_pos = np.sin(angles)  # shape: (max_pos, d//2)

# Interleave cos and sin to form full rotary matrix (approx)
# This shows how the embedding changes per position
rope_pattern = np.zeros((max_pos, d))
rope_pattern[:, 0::2] = cos_pos
rope_pattern[:, 1::2] = sin_pos

# Plot the cosine and sine components
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cos_pos, aspect='auto', cmap='viridis')
plt.colorbar(label='Cos(m * theta_i)')
plt.xlabel('Dimension Pairs (i)')
plt.ylabel('Position (m)')
plt.title('Cosine Components of RoPE')

plt.subplot(1, 2, 2)
plt.imshow(sin_pos, aspect='auto', cmap='viridis')
plt.colorbar(label='Sin(m * theta_i)')
plt.xlabel('Dimension Pairs (i)')
plt.ylabel('Position (m)')
plt.title('Sine Components of RoPE')

plt.tight_layout()
plt.show()

# Optional: Plot how a single embedding rotates over positions
plt.figure(figsize=(8, 8))
for pos in [0, 5, 10, 20, 30, 40]:
    x = cos_pos[pos, 0]  # First 2D component
    y = sin_pos[pos, 0]
    plt.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, label=f'Pos {pos}' if pos in [0, 40] else "")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('cos(m*θ)')
plt.ylabel('sin(m*θ)')
plt.title('2D Rotation of First Frequency Over Positions')
plt.legend()
plt.axis('equal')
plt.show()