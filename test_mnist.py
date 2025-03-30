import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras_visualizer import visualizer


# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Print dataset shapes before preprocessing
print("Before preprocessing:")
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Data type: {x_train.dtype}")
print(f"Min value: {x_train.min()}, Max value: {x_train.max()}")

# Normalize the data (divide by 255 to get values between 0 and 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten the images from 28x28 to 784x1
x_train_flattened = x_train.reshape(x_train.shape[0], -1)
x_test_flattened = x_test.reshape(x_test.shape[0], -1)

# One-hot encode the labels
num_classes = 10  # digits 0-9
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)

# Print dataset shapes after preprocessing
print("\nAfter preprocessing:")
print(f"Training data shape (flattened): {x_train_flattened.shape}")
print(f"Training labels shape (one-hot): {y_train_onehot.shape}")
print(f"Test data shape (flattened): {x_test_flattened.shape}")
print(f"Test labels shape (one-hot): {y_test_onehot.shape}")

# Visualize original labels vs one-hot encoded labels
print("\nOriginal labels (first 5):")
print(y_train[:5])

print("\nOne-hot encoded labels (first 5):")
for i in range(5):
    print(f"Digit {y_train[i]}: {y_train_onehot[i]}")

# Visualize an example image and its one-hot encoded label
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Original Label: {y_train[0]}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(range(10), y_train_onehot[0])
plt.xticks(range(10))
plt.title('One-Hot Encoded Label')
plt.xlabel('Digit Class')
plt.ylabel('Value (0 or 1)')

plt.tight_layout()
plt.show()

# Visualize the effect of flattening
plt.figure(figsize=(12, 5))

# Original 2D image
plt.subplot(1, 2, 1)
plt.imshow(x_train[0], cmap='gray')
plt.title('Original 2D Image (28×28)')
plt.axis('off')

# Flattened image reshaped for visualization
plt.subplot(1, 2, 2)
plt.imshow(x_train_flattened[0].reshape(28, 28), cmap='gray')
plt.title('Flattened then Reshaped (784 → 28×28)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Visualize the flattened image as a 1D array
plt.figure(figsize=(12, 3))
plt.plot(x_train_flattened[0])
plt.title('Flattened Image as 1D Array (784 values)')
plt.xlabel('Pixel Position')
plt.ylabel('Pixel Value (0-1)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize a grid of examples (normalized)
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Display a histogram of pixel values before and after normalization
plt.figure(figsize=(12, 5))

# Create a sample of original data for the histogram
original_sample = np.random.choice(x_train.shape[0], 1000)
original_pixels = tf.keras.datasets.mnist.load_data()[0][0][original_sample].flatten()

plt.subplot(1, 2, 1)
plt.hist(original_pixels, bins=50)
plt.title('Original Pixel Values')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(x_train[original_sample].flatten(), bins=50)
plt.title('Normalized Pixel Values')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# ================ NEURAL NETWORK MODEL ================
# Alternative imports


# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # hidden layer 1
    Dense(64, activation='relu'),                       # hidden layer 2
    Dense(10, activation='softmax')                     # output layer (10 digits)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Use categorical_crossentropy with one-hot encoded labels
    metrics=['accuracy']
)

# View the model structure
model.summary()

# Optional: Visualize the model architecture
print("\nModel Architecture:")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}, Output Shape: {layer.output.shape}, Parameters: {layer.count_params()}")

# Visualize the model architecture using keras_visualizer
visualizer(model)

# Create a more detailed visualization showing some neurons and connections
plt.figure(figsize=(15, 10))

# Define parameters
n_layers = 4
layer_sizes = [784, 128, 64, 10]
layer_names = ['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer']
max_neurons_to_show = [10, 10, 10, 10]  # Show at most this many neurons per layer
neuron_radius = 0.25
layer_spacing = 3

# Colors
layer_colors = ['#FFC107', '#4CAF50', '#2196F3', '#F44336']
connection_color = 'gray'

# Calculate positions
x_positions = [i * layer_spacing for i in range(n_layers)]
y_ranges = [range(min(size, max_neurons_to_show[i])) for i, size in enumerate(layer_sizes)]

# Draw connections (only a subset for visibility)
for i in range(n_layers - 1):
    for j in range(len(y_ranges[i])):
        if j >= max_neurons_to_show[i]:
            break
        y1 = j - len(y_ranges[i]) / 2
        
        # Only connect to a subset of neurons in the next layer
        for k in range(min(3, len(y_ranges[i+1]))):
            y2 = k - len(y_ranges[i+1]) / 2
            plt.plot([x_positions[i], x_positions[i+1]], [y1, y2], 
                     color=connection_color, alpha=0.1, linewidth=0.5)

# Draw neurons
for i in range(n_layers):
    x = x_positions[i]
    
    # Draw visible neurons
    for j in range(min(layer_sizes[i], max_neurons_to_show[i])):
        y = j - min(layer_sizes[i], max_neurons_to_show[i]) / 2
        circle = plt.Circle((x, y), neuron_radius, color=layer_colors[i], alpha=0.7)
        plt.gca().add_patch(circle)
    
    # If we're not showing all neurons, add ellipsis
    if layer_sizes[i] > max_neurons_to_show[i]:
        plt.text(x, -min(layer_sizes[i], max_neurons_to_show[i])/2 - 1, "...", 
                 ha='center', va='center', fontsize=20)
        plt.text(x, min(layer_sizes[i], max_neurons_to_show[i])/2 + 1, "...", 
                 ha='center', va='center', fontsize=20)
    
    # Add layer labels
    plt.text(x, -min(layer_sizes[i], max_neurons_to_show[i])/2 - 2.5, 
             f"{layer_names[i]}\n({layer_sizes[i]} neurons)", 
             ha='center', va='center', fontsize=12)

# Set plot limits and remove axes
plt.xlim(min(x_positions) - 2, max(x_positions) + 2)
plt.ylim(-max([len(r) for r in y_ranges]) - 3, max([len(r) for r in y_ranges]) + 3)
plt.axis('off')
plt.title('Neural Network Architecture (Simplified View)', fontsize=16)
plt.tight_layout()
plt.show()

# ================ TRAIN THE MODEL ================
# Train the model on the flattened images and one-hot encoded labels
history = model.fit(
    x_train_flattened,           # Input: flattened images
    y_train_onehot,              # Target: one-hot encoded labels
    epochs=5,                    # Number of passes through the entire dataset
    batch_size=32,               # Number of samples per gradient update
    validation_split=0.1,        # Use 10% of training data as validation
    verbose=1                    # Show progress bar
)

# Plot the training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ================ EVALUATE THE MODEL ================
# Evaluate on test data
test_loss, test_accuracy = model.evaluate(
    x_test_flattened,  # Use flattened test images
    y_test_onehot,     # Use one-hot encoded test labels
    verbose=1
)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Make predictions
predictions = model.predict(x_test_flattened)

# Visualize a few predictions
plt.figure(figsize=(15, 8))

# Function to show prediction details
def plot_prediction(index, position):
    plt.subplot(2, 5, position)
    plt.imshow(x_test[index], cmap='gray')
    
    predicted_label = np.argmax(predictions[index])
    actual_label = y_test[index]
    
    # Set title color based on correctness
    title_color = 'green' if predicted_label == actual_label else 'red'
    
    plt.title(f"Pred: {predicted_label}, Actual: {actual_label}", 
              color=title_color, fontweight='bold')
    plt.axis('off')
    
    # Add prediction probabilities as a bar chart
    plt.subplot(2, 5, position+5)
    bars = plt.bar(range(10), predictions[index], color='blue', alpha=0.7)
    
    # Highlight the predicted and actual classes
    bars[predicted_label].set_color('green')
    if predicted_label != actual_label:
        bars[actual_label].set_color('red')
    
    plt.xticks(range(10))
    plt.ylim(0, 1)
    plt.title('Class Probabilities')

# Show predictions for 5 random test images
random_indices = np.random.choice(len(x_test), 5, replace=False)
for i, idx in enumerate(random_indices):
    plot_prediction(idx, i+1)

plt.tight_layout()
plt.show()

# Show a confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Convert predictions to class labels
y_pred = np.argmax(predictions, axis=1)
y_true = y_test

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Find and display some misclassified examples
misclassified = np.where(y_pred != y_true)[0]
print(f"\nNumber of misclassified images: {len(misclassified)} out of {len(y_test)} test images")

# Show a few misclassified examples
plt.figure(figsize=(15, 4))
for i in range(min(5, len(misclassified))):
    idx = misclassified[i]
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}, True: {y_true[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

def predict_custom_image(image_path):
    # Load and preprocess the image
    from PIL import Image, ImageOps
    import numpy as np
    
    # Open image and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Invert colors if needed (MNIST has white digits on black background)
    img = ImageOps.invert(img)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Display the processed image
    plt.figure(figsize=(4, 4))
    plt.imshow(img_array, cmap='gray')
    plt.title("Processed Input Image")
    plt.axis('off')
    plt.show()
    
    # Flatten and make prediction
    img_flattened = img_array.reshape(1, 784)  # Add batch dimension
    prediction = model.predict(img_flattened)
    
    # Get predicted digit and confidence
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit] * 100
    
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Show prediction probabilities
    plt.figure(figsize=(10, 4))
    plt.bar(range(10), prediction[0])
    plt.xticks(range(10))
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.show()

# Usage:
# predict_custom_image('path/to/your/image.jpg')

# Add this at the end of your script to test your image
predict_custom_image('/Users/zachderhake/Downloads/1709231338354.jpeg')