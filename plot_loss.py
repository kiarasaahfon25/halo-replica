import pickle
import matplotlib.pyplot as plt

# Load saved losses
with open('./save/losses.pkl', 'rb') as f:
    losses = pickle.load(f)

train_losses = losses['train_losses']
val_losses = losses['val_losses']

# Epoch numbers
train_epochs = range(1, len(train_losses) + 1)
val_epochs = range(1, len(val_losses) + 1)

# Plot training and validation loss
plt.plot(train_epochs, train_losses, label='Training Loss')
plt.plot(val_epochs, val_losses, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('HALO Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.savefig('./save/loss_curve.png')
plt.show()
