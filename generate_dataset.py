import numpy as np
from PIL import Image
import json
import pickle
from simple_deep_learning.mnist_extended.semantic_segmentation import create_semantic_segmentation_dataset

# Let's generate 30,000 training images and 5,000 test images

kwargs = {
    'num_train_samples': 30000,
    'num_test_samples': 5000,
    'image_shape': (60, 60),
    'min_num_digits_per_image': 3,
    'max_num_digits_per_image': 3,
    'num_classes': 5,
    'max_iou': 0.1,
    'duplicate_digits': False
}

train_x, train_y, train_z, test_x, test_y, test_z = create_semantic_segmentation_dataset(**kwargs)

# Save training images
for i in range(len(train_x)):
    grayscale_image = (train_x[i] * 255).astype(np.uint8)
    grayscale_image = np.squeeze(grayscale_image, axis=2)
    pil_image = Image.fromarray(grayscale_image, mode='L')
    pil_image.save(f'/users/bspiegel/data/bspiegel/extended-mnist/dataset1/train/image_{i}.png')

# Save testing images
for i in range(len(test_x)):
    grayscale_image = (test_x[i] * 255).astype(np.uint8)
    grayscale_image = np.squeeze(grayscale_image, axis=2)
    pil_image = Image.fromarray(grayscale_image, mode='L')
    pil_image.save(f'/users/bspiegel/data/bspiegel/extended-mnist/dataset1/test/image_{i}.png')

# Save generation args as json file
file_path = '/users/bspiegel/data/bspiegel/extended-mnist/dataset1/config.json'  # Specify the path and filename for the JSON file
with open(file_path, 'w') as json_file:
    json.dump(kwargs, json_file)

# Pickle z labels
file_path = '/users/bspiegel/data/bspiegel/extended-mnist/dataset1/train_labels.pkl'  # Specify the path and filename for the pickle file
with open(file_path, 'wb') as file:
    pickle.dump(train_z, file)
file_path = '/users/bspiegel/data/bspiegel/extended-mnist/dataset1/test_labels.pkl'  # Specify the path and filename for the pickle file
with open(file_path, 'wb') as file:
    pickle.dump(test_z, file)


# Visualize y labels

# original_array = train_y[0]
# split_arrays = np.split(original_array, 5, axis=2)
# # Verify the shapes of the split arrays
# for i, array in enumerate(split_arrays):
#     print(f"Split array {i+1}: {array.shape}")
#     grayscale_image = (array * 255).astype(np.uint8)
#     # Remove the extra dimension
#     grayscale_image = np.squeeze(grayscale_image, axis=2)
#     # Convert NumPy array to PIL image
#     pil_image = Image.fromarray(grayscale_image, mode='L')
#     # Save the image
#     pil_image.save(f'grayscale_image_{i}.png')