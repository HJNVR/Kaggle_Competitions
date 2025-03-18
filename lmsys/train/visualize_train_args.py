train_log_losses = [0.4009885966017323, 0.383720020641981, 0.37854219339290374, 0.3516045669728663, 0.34647175705802585, 0.33898693967794774, 0.3188461911056889, 0.3023486988119195, 0.2796152251866029, 0.254708720240542]
val_log_losses = [0.3799550106500586, 0.368797197163105, 0.40669376309368854, 0.36212081918502464, 0.34732329039719817, 0.34951043998526454, 0.3569389301917922, 0.37307021735849766, 0.3941470826448013, 0.41047134168639493]
tr_accuracies = [0.8463256746900073, 0.8515226112326769, 0.8573577680525164, 0.8616885485047411, 0.862281181619256, 0.8626458789204959, 0.8682986870897156, 0.8758205689277899, 0.8869894237782641, 0.8949215900802334]
val_accuracies= [0.8528642494486897, 0.8583004256628545, 0.85035130006667, 0.8576337248064003, 0.8619416380327196, 0.8635314631519565, 0.8619929227139853, 0.8554797681932407, 0.8527616800861583, 0.8428124519206113]

import matplotlib.pyplot as plt
import os
# Plotting both log losses and accuracies in subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot training and validation log loss
ax1.plot(train_log_losses, label='Train Log Loss')
ax1.plot(val_log_losses, label='Validation Log Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Log Loss')
ax1.legend()
ax1.set_title('Training and Validation Log Loss')

# Plot training and validation accuracy
ax2.plot(tr_accuracies, label='Train Accuracy')
ax2.plot(val_accuracies, label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.set_title('Training and Validation Accuracy')

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join('/data0/huangjing/workspace/kaggle/lmsys/ddp/loss_history',f'epoch_{5}.png'))

# Show the plot
plt.show()