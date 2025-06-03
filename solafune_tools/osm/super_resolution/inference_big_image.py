import cv2
import numpy as np
from inference import Model as SRModel

def split_image(image, patch_size):
    height, width = image.shape[:2]
    patches = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches

def combine_patches(patches, output_size):
    output = np.zeros(output_size, dtype=np.float32)
    count = np.zeros(output_size, dtype=np.float32)
    for patch, (y, x) in patches:
        output[y:y+patch.shape[0], x:x+patch.shape[1]] += patch
        count[y:y+patch.shape[0], x:x+patch.shape[1]] += 1
    output /= count
    return output.astype(np.uint8)

def apply_model_to_image(image, model):
    patch_size = 650
    patches = split_image(image, patch_size)
    processed_patches = []
    for patch in patches:
        model = SRModel
        
    output_size = (image.shape[0], image.shape[1], processed_patches[0][0].shape[2])
    output_image = combine_patches(processed_patches, output_size)
    return output_image

# Load the large image
image = cv2.imread('test_data_bucket/mango.jpg')


# Apply the model to the image
output_image = apply_model_to_image(image, model)

# Save the output image
cv2.imwrite('path/to/output/image.jpg', output_image)