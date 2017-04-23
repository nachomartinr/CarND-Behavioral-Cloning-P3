import argparse
import base64
from datetime import datetime
import os
import shutil
from io import BytesIO
import h5py
import eventlet.wsgi
import numpy as np
import socketio
import tensorflow as tf
from PIL import Image
from flask import Flask
from keras.models import load_model
from keras import __version__ as keras_version
from scipy.misc import imresize

tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


IMAGE_SIZE_Y = 64
IMAGE_SIZE_X = 64


def preprocess_crop(image):
    """
    Crop and image removing the the part above the horizon from the top and the hood of the car from the botom.     
    """
    top_horizon = int(np.ceil(image.shape[0] * 0.3))
    bottom_hood = image.shape[0] - int(np.ceil(image.shape[0] * 0.1))

    return image[top_horizon:bottom_hood, :]


def preprocess_resize(image):
    """
    Resize the image sample to the target input size for the CNN.    
    """
    # scipy.misc.imresize (which uses PIL) is called instead of cv2.resize, as the latter
    # was causing CUBLAS errors in my machine when running it in drive.py.
    
    # scipy.misc.imresize uses (height, width) for size (whereas PIL and CV2 use (width, height))    
    image = imresize(image, (IMAGE_SIZE_Y, IMAGE_SIZE_X), interp='bilinear')
    return image

def preprocess_image(image):
    image = preprocess_crop(image)
    image = preprocess_resize(image)
    return image

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.05, 0.001)
set_speed = 30
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]

    # The current throttle of the car
    throttle = data["throttle"]

    # The current speed of the car
    speed = data["speed"]

    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    image_array = preprocess_image(image_array)

    transformed_image_array = image_array[None, :, :, :]

    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    #throttle = 0.3
    throttle = controller.update(float(speed))

    print('{:.5f}, {:.1f}'.format(steering_angle, throttle))

    send_control(steering_angle, throttle)
    
    # save frame
    if args.image_folder != '':
        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        image_filename = os.path.join(args.image_folder, timestamp)
        image.save('{}.jpg'.format(image_filename))


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)
    
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

