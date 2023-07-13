import os
import time
import tensorflow as tf


def process_images(image, label):
    """ 
       Normalize image to have a mean of 0 and a standard deviation of 1
       and a size of 277x277
    """
    # Normalizes the image
    image = tf.image.per_image_standardization(image)
    # Resize image to 277x277
    image = tf.image.resize(image, (277, 277))
    return image, label


root_logdir = os.path.join(os.curdir, "logs//fit//")


def get_run_logdir():
    """
    Returns a reference to the directory to store TensorBoard files
    """
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
