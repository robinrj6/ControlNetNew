# plot the sudden convergence of the training loss for the controlnet model trained on the coco dataset. The training loss is obtained from the logs folder .err file. The plot should show the loss values on the y-axis and the iteration number on the x-axis. The plot should also have a title and labels for the axes.

import matplotlib.pyplot as plt
import re
import os
import pandas as pd
import argparse

# ===== CONFIG =====
parser = argparse.ArgumentParser(description='Plot training loss convergence from log files')
parser.add_argument('fileName', type=str, help='Log file name (without .err extension)')
parser.add_argument('--plot-type', type=str, default='both', choices=['raw', 'smoothed', 'both'],
                    help='Type of plot: raw, smoothed, or both (default: both)')
parser.add_argument('--window', type=int, default=50, 
                    help='Rolling average window size (default: 50)')
args = parser.parse_args()

fileName = args.fileName
plot_type = args.plot_type
smoothing_window = args.window
# ==================
# Read the training loss values from the file
with open(f'logs/{fileName}.err', 'r') as f:
    lines = f.readlines()

loss_values = []
steps = []

for line in lines:
    if 'loss=' in line:
        # Extract loss value
        loss_match = re.search(r'loss=([0-9.]+)', line)
        if loss_match:
            loss_values.append(float(loss_match.group(1)))
        
        # Extract steps (current/total format like 902/12500)
        steps_match = re.search(r'\|.*?\|\s+(\d+)/\d+', line)
        if steps_match:
            steps.append(int(steps_match.group(1)))

# Filter to only loss < 1
filtered_steps = []
filtered_loss = []
for step, loss in zip(steps, loss_values):
    if loss < 1:
        filtered_steps.append(step)
        filtered_loss.append(loss)

print(f"Original data points: {len(loss_values)}")
print(f"Filtered data points (loss < 1): {len(filtered_loss)}")

# Apply rolling average smoothing (50-step window)
loss_series = pd.Series(filtered_loss)
smoothed_loss = loss_series.rolling(smoothing_window, center=True).mean()

# Plot the training loss
plt.figure(figsize=(12, 6))

if plot_type == "raw":
    plt.plot(filtered_steps, filtered_loss, linewidth=2, color='blue', label='Raw loss')
elif plot_type == "smoothed":
    plt.plot(filtered_steps, smoothed_loss, linewidth=2, marker='o', markersize=3, color='blue', label=f'Smoothed loss ({smoothing_window}-step MA)')
elif plot_type == "both":
    plt.plot(filtered_steps, filtered_loss, linewidth=0.5, alpha=0.3, label='Raw loss', color='lightblue')
    plt.plot(filtered_steps, smoothed_loss, linewidth=2, marker='o', markersize=3, label=f'Smoothed loss ({smoothing_window}-step MA)', color='blue')

plt.title('Training Loss Convergence (loss < 1)', fontsize=14)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.gca().invert_yaxis()  # Invert y-axis: highest loss at top, 0 at bottom
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Create results folder if it doesn't exist
os.makedirs('results', exist_ok=True)

# Save the figure
plt.savefig(f'results/{fileName}.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to results/{fileName}.png")
plt.show()