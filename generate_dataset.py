import numpy as np
from PIL import Image
import json
import pickle
import argparse
from simple_deep_learning.mnist_extended.semantic_segmentation import create_semantic_segmentation_dataset

# Let's generate 30,000 training images and 5,000 test images

def label_filter_no_04(labels):
    if 0 in labels and 4 in labels:
        return False
    else:
        return True

def label_filter_only_04(labels):
    if 0 in labels and 4 in labels:
        return True
    else:
        return False

def label_filter_fn_generator(include=None, exclude=None):
    def generated_filter(labels):
        if include is not None:
            for label in include:
                if label not in labels:
                    return False
        if exclude is not None:
            is_included = None
            for label in exclude:
                if label in labels:
                    if is_included is None:
                        is_included = True
                    else:
                        is_included = is_included and True
                else:
                    if is_included is None:
                        is_included = False
                    else:
                        is_included = is_included and False
            return not is_included
        return True
    return generated_filter


def generate(dataset_name, condition_a_label_filter_function, condition_b_label_filter_function, kwargs):
    # Save generation args as json file
    file_path = f'/users/bspiegel/data/bspiegel/extended-mnist/{dataset_name}/config.json'  # Specify the path and filename for the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(kwargs, json_file)

    del kwargs['dataset_name']
    del kwargs['exclude']

    train_a_x, train_a_y, train_a_z, test_a_x, test_a_y, test_a_z, test_b_x, test_b_y, test_b_z = create_semantic_segmentation_dataset(**kwargs, condition_a_label_filter_function=condition_a_label_filter_function, condition_b_label_filter_function=condition_b_label_filter_function)

    # Save training images
    for i in range(len(train_a_x)):
        grayscale_image = (train_a_x[i] * 255).astype(np.uint8)
        grayscale_image = np.squeeze(grayscale_image, axis=2)
        pil_image = Image.fromarray(grayscale_image, mode='L')
        pil_image.save(f'/users/bspiegel/data/bspiegel/extended-mnist/{dataset_name}/train/image_{i}.png')

    # Save testing images
    for i in range(len(test_a_x)):
        grayscale_image = (test_a_x[i] * 255).astype(np.uint8)
        grayscale_image = np.squeeze(grayscale_image, axis=2)
        pil_image = Image.fromarray(grayscale_image, mode='L')
        pil_image.save(f'/users/bspiegel/data/bspiegel/extended-mnist/{dataset_name}/testa/image_{i}.png')

    for i in range(len(test_b_x)):
        grayscale_image = (test_b_x[i] * 255).astype(np.uint8)
        grayscale_image = np.squeeze(grayscale_image, axis=2)
        pil_image = Image.fromarray(grayscale_image, mode='L')
        pil_image.save(f'/users/bspiegel/data/bspiegel/extended-mnist/{dataset_name}/testb/image_{i}.png')

    # Pickle z labels
    file_path = f'/users/bspiegel/data/bspiegel/extended-mnist/{dataset_name}/train_labels.pkl'  # Specify the path and filename for the pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(train_a_z, file)
    file_path = f'/users/bspiegel/data/bspiegel/extended-mnist/{dataset_name}/test_a_labels.pkl'  # Specify the path and filename for the pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(test_a_z, file)
    file_path = f'/users/bspiegel/data/bspiegel/extended-mnist/{dataset_name}/test_b_labels.pkl'  # Specify the path and filename for the pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(test_b_z, file)


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

# You may need to run this command if tensorflow is not seeing CUDA:
# module load cuda/11.7.1 cudnn/8.2.0
#
# Verify CUDA with:
# tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extended MNIST dataset generation tool.')
    parser.add_argument('--dataset_name', type=str, default='exclude13b', help='Name of the dataset to generate.', required=True)
    parser.add_argument('--exclude', type=list, default=None, help='List of digits to exclude from the training dataset and test a dataset, to be put into the test b dataset.', required=True)

    parser.add_argument('--duplicate_digits', type=bool, default=False, help='Allow duplicate digits in the same image.', required=False)
    parser.add_argument('--max_iou', type=float, default=0.1, help='Maximum IOU between digits.', required=False)
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes to generate.', required=False)
    parser.add_argument('--min_num_digits_per_image', type=int, default=3, help='Minimum number of digits per image.', required=False)
    parser.add_argument('--max_num_digits_per_image', type=int, default=3, help='Maximum number of digits per image.', required=False)    
    parser.add_argument('--image_shape', type=tuple, default=(60, 60), help='Shape of the images to generate.', required=False)
    parser.add_argument('--num_train_samples', type=int, default=30000, help='Number of training samples to generate.', required=False)
    parser.add_argument('--num_test_samples', type=int, default=5000, help='Number of testing samples to generate.', required=False)
    parser.add_argument('--num_test_b_samples', type=int, default=5000, help='Number of testing samples to generate.', required=False)

    args = vars(parser.parse_args())

    # kwargs = {
    #     'num_train_samples': 30000,
    #     'num_test_samples': 5000,
    #     'num_test_b_samples': 5000,
    #     'image_shape': (60, 60),
    #     'min_num_digits_per_image': 3,
    #     'max_num_digits_per_image': 3,
    #     'num_classes': 5,
    #     'max_iou': 0.1,
    #     'duplicate_digits': False
    # }

    non_json_kwargs = dict()

    # Generate the appropriate filter functions according to the exclude list
    if args['exclude'] is not None:
        args['exclude'] = [int(x) for x in args['exclude']]
        non_json_kwargs['condition_a_label_filter_function'] = label_filter_fn_generator(exclude=args['exclude'])
        non_json_kwargs['condition_b_label_filter_function'] = label_filter_fn_generator(include=args['exclude'])
    else:
        non_json_kwargs['condition_a_label_filter_function'] = lambda x: True
        non_json_kwargs['condition_b_label_filter_function'] = lambda x: True

    # print(args)
    # print(args['dataset_name'], non_json_kwargs)
    generate(args['dataset_name'], **non_json_kwargs, kwargs=args)
    
