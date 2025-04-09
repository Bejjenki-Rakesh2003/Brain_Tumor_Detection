import matplotlib.pyplot as plt
import json

# Load training history
with open("D:/brain_tumor_detection/training_history.json", "r") as f:
    history = json.load(f)

# Create figure and subplots
plt.figure(figsize=(12, 5))

# Accuracy Plot (Left)
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot (Right)
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save combined figure
plt.tight_layout()
plt.savefig('D:/brain_tumor_detection/model_performance.png')
plt.show()
