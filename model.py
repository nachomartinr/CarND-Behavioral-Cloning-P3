import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Lambda, Convolution2D, Activation, Flatten, Dense, Dropout 
from keras.optimizers import Adam
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from scipy.misc import imresize
import tensorflow as tf
tf.python.control_flow_ops = tf


# Some useful constants
IMAGE_SIZE_Y = 64
IMAGE_SIZE_X = 64
IMAGE_NUM_CH = 3
IMAGE_SIZE = (IMAGE_SIZE_Y, IMAGE_SIZE_X, IMAGE_NUM_CH)

NUM_EPOCHS=7
SAMPLES_PER_EPOCH=20000
BATCH_SIZE=200

DRIVING_LOG_PATH = './data/driving_log.csv'
IMG_PATH = './data/'


def new_model():
    """
    Create a new CNN model     
    """
    
    row, col, ch = IMAGE_SIZE_Y, IMAGE_SIZE_X, IMAGE_NUM_CH
    
    p = 0.2
    
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch), name='normalization'))
    
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), name='conv1'))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), name='conv2'))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), name='conv3'))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), name='conv4'))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), name='conv5'))
    model.add(Activation('relu'))

    model.add(Flatten())

    # Next, five fully connected layers
    model.add(Dense(1164, name="dense1"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(100, name="dense2"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(50, name="dense3"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(10, name="dense4"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(1, name="output"))

    model.summary()

    return model


def my_load_model():
    model = load_model('model.h5')
    return model


def save_model(model):
    model.save('model.h5')  # creates a HDF5 file 'model.h5'
    print("Model Saved")


def train_model(fine_tune=False, learning_rate=0.0001):
    """
    Load the data, split the training/validation set and train the model.     
    """
    if fine_tune:
        model = my_load_model() # load previously stored model (from 'model.h5')
    else:
        model = new_model() # create new model
        
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
                  
    data = pd.read_csv(DRIVING_LOG_PATH, names=None) # Load the data using Pandas.
    X_train = data[['left','center','right']].to_dict('records') # each data sample contains the paths to three camera images.
    y_train = data['steering'].as_matrix().astype(np.float32) # steering data

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
                  
    
    train_generator = train_batch_generator(X_train, y_train, BATCH_SIZE)
    X_val_prep, y_val_prep = preprocess_validation_data(X_val, y_val)
        
    history = model.fit_generator(train_generator,
                                  samples_per_epoch=SAMPLES_PER_EPOCH, nb_epoch=NUM_EPOCHS,
                                  validation_data=(X_val_prep, y_val_prep),
                                  verbose=1)
         
    save_model(model)


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


def augment_flip(image, steering_angle):
    """
    Flip the image with a probability of 0.5.
    The steering is negated for flipped samples.
    """ 
    if np.random.randint(2) == 0:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
   
    return image, steering_angle


def augment_brightness(image):
    """
    Change the brightness of training images using gamma correction.
    http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction
    """
    gamma = np.random.uniform(0.4, 1.5) #randomly select a gamma value
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def augment_shear(image, steering_angle):
    """
    Apply random horizontal shear between [-200, 201].
    The probability of applying shear is 90%
    https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.tg2ah5ebl
    
    """
   
    if np.random.random() < 0.9: # apply shear with a probability of 90%
        rows, cols, _ = image.shape
        dx = np.random.randint(-200, 201) # get a random displacement
        random_c_point = [cols/2 + dx, rows/2] # get a random point horizontally displaced from the center
        # Coordinates of triangle vertices in the source image.
        pts1 = np.float32([[0,rows],[cols,rows],[cols/2, rows/2]])
        # Coordinates of triangle vertices in the destination image.
        pts2 = np.float32([[0,rows],[cols,rows],random_c_point])
        # Calculate the steering correction
        dsteering = dx/(rows/2) * 360/(2*np.pi*25.0)/6.0
        # Calculates the affine transform matrix from the points of two corresponding triangles.
        Mtx = cv2.getAffineTransform(pts1, pts2)
        # Transform the image
        image = cv2.warpAffine(image, Mtx, (cols, rows), borderMode=1)
        steering_angle += dsteering

    return image, steering_angle


def preprocess_training_sample(image, steering_angle):
    """
    Apply the preprocessin pipeline. 
    1- Perform random shearing for data augmentation
    2- Crop the image to remove useless parts of the image.
    3- Peform flip with a probability of 50% for data augmentation
    4- Perform random brighness change for data augmentation
    5- Change color space from BGR to RGB as drive.py loads images in RGB.
    6- Resize images to target size to feed the CNN.
    """
    image, steering_angle = augment_shear(image, steering_angle)
    image = preprocess_crop(image)    
    image, steering_angle = augment_flip(image, steering_angle)
    image = augment_brightness(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_resize(image)   
    
    return image, steering_angle


def get_training_sample(X_train, y_train):    
    """
    Get a new augmented training sample.
    """ 
    
    # Randomly select the sample index within all the dataset.
    index = np.random.randint(0, len(y_train))
    
    # Randomly select camera (left, center, right) the correspoding steering angle correction
    lrc = np.random.randint(3)   
    if (lrc == 0):
        img_path = X_train[index]['left'].strip()
        off_angle = 0.23
    elif (lrc == 1):
        img_path = X_train[index]['center'].strip()
        off_angle = 0.
    elif (lrc == 2):
        img_path = X_train[index]['right'].strip()
        off_angle = -0.23
        
    image = cv2.imread(IMG_PATH + img_path) # Load the selected image

    steering_angle = y_train[index] # Get the angle for the center image
    steering_angle += off_angle # Apply angle correction
    
    # Preprocess and randomly augment the sample
    image, steering_angle = preprocess_training_sample(image, steering_angle)
    
    return image, steering_angle

   
def train_batch_generator(X_train, y_train, batch_size=200):    
    """
    Create batches of training samples.
    """ 
 
    while True:
        X_batch = np.ndarray(shape=(batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), dtype=float)
        y_batch = np.ndarray(shape=(batch_size), dtype=float)
        for i in range(batch_size):
            #  Get a new augmented training sample.
            image, steering_angle = get_training_sample(X_train, y_train)                
            
            # Add sample to the current batch
            X_batch[i] = image
            y_batch[i] = steering_angle
            

        yield X_batch, y_batch


def preprocess_validation_sample(image):
    """
    Apply crop, conversion to RGB and resize to a validation sample.
    Preprocessing steps must be the same as in the traning samples, but with no agumentation
    """ 
    image = preprocess_crop(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_resize(image)
    return image    
        
        
def get_validation_sample(X_val, y_val, index):    
    """
    Get the validation sample indicated by index and preprocess it.
    """ 
    img_path = X_val[index]['center'].strip()        
    image = cv2.imread(IMG_PATH + img_path)
    steering_angle = y_val[index]
       
    image = preprocess_validation_sample(image)
    
    return image, steering_angle 


def preprocess_validation_data(X_val, y_val):
    """
    Get the validation data set with preprocessed images
    """ 
    X_val_prep = np.ndarray(shape=(len(X_val), IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), dtype=float)    
    y_val_prep = np.ndarray(shape=(len(y_val)), dtype=float)
    for i in range(len(X_val)):
        image, steering_angle = get_validation_sample(X_val, y_val, i)
        X_val_prep[i] = image.astype(float)
        y_val_prep[i] = steering_angle
        
    return X_val_prep, y_val_prep
    
if __name__ == '__main__':
    train_model()

