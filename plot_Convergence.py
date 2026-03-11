# plot the sudden convergence of the training loss for the controlnet model trained on the coco dataset. The training loss is obtained from the logs folder .err file. The plot should show the loss values on the y-axis and the iteration number on the x-axis. The plot should also have a title and labels for the axes.

import matplotlib.pyplot as plt
import re
import os

fileName = "controlnet_train_Fill50k_1548404"
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

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(filtered_steps, filtered_loss, linewidth=2)
plt.title('Training Loss Convergence (loss < 1)', fontsize=14)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.gca().invert_yaxis()  # Invert y-axis: highest loss at top, 0 at bottom
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Create results folder if it doesn't exist
os.makedirs('results', exist_ok=True)

# Save the figure
plt.savefig('results/training_loss_convergence.png', dpi=300, bbox_inches='tight')
print("Plot saved to results/training_loss_convergence.png")
plt.show()