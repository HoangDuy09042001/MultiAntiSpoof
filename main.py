from fake_face_dectection import predict
from PIL import Image
import numpy as np

def image_to_numpy(image_path):
    # Open the image file
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    return image_array
image_path = 'anhtest.jpg'
image_array = image_to_numpy(image_path)

blinding_light = {
    "image_arrays": [image_array,image_array,image_array,image_array, image_array, image_array, image_array]
}
predict(blinding_light)