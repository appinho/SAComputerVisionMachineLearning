import matplotlib.image as mpimg
import cv2
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,MaxPooling2D,Cropping2D
import matplotlib.pyplot as plt

# load data
def load_data():
    # define path
    directory = '../Recorded_Data/'
    data_paths = ['Loop_Data/', 'Loop_Data_Reversed/',
                  'Recover_Data/', 'Recover_Data_Reversed/',
                  'Bridge_Recover_Data/', 'Curve_Recover_Data/',
                  'Recover_Data_2/', 'Loop_Data_2/']
    data_usage = [True, True, True, True, True, True, True, True]
    data_augmentation = True
    use_multiple_cameras = True
    correction_factor = 0.2

    # init image and measurement buffer
    images = []
    measurements = []

    # open csv file from simulation data
    for counter, data_path in enumerate(data_paths):
        if not data_usage[counter]:
            continue
        print(directory + data_path + "loading..")
        # init csv file buffer
        lines = []

        with open(directory + data_path + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

        # loop through frames and store images and measurements
        for line in lines:
            source_path = line[0]
            file_name = source_path.split('/')[-1]
            current_path = directory + data_path + 'IMG/' + file_name
            image = mpimg.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)
            if use_multiple_cameras:
                if not line[1] == "":
                    source_path = line[1]
                    file_name = source_path.split('/')[-1]
                    current_path = directory + data_path + 'IMG/' + file_name
                    image = mpimg.imread(current_path)
                    images.append(image)
                    measurement = float(line[3]) + correction_factor
                    measurements.append(measurement)
                if not line[2] == "":
                    source_path = line[2]
                    file_name = source_path.split('/')[-1]
                    current_path = directory + data_path + 'IMG/' + file_name
                    image = mpimg.imread(current_path)
                    images.append(image)
                    measurement = float(line[3]) - correction_factor
                    measurements.append(measurement)

    # data augmentation
    if data_augmentation:
        print("Before data augmentation = " + str(len(images)))
        # determine
        number_of_images = len(images)
        for i in range(number_of_images):
            # flipped data
            images.append(cv2.flip(images[i], 1))
            measurements.append(measurements[i] * -1.0)

        print("After data augmentation = " + str(len(images)))

    # Define training data
    X_train = np.array(images)
    y_train = np.array(measurements)

    print('Shape image data: ', X_train.shape)
    print('Shape measurement data: ', y_train.shape)

    # Plot training set
    plt.hist(y_train, 29, facecolor='g', alpha=0.75)
    plt.title('Histogram of steering angles')
    plt.grid(True)
    plt.show()

    # Return dataset
    return X_train,y_train

# define the network architecture
def nvidia_net(X_train,y_train,num_epochs,model_name):
    # Read input image shape
    input_shape = X_train.shape[1:4]

    # Define sequential model
    model = Sequential()

    # Preprocessing 1: Cropping the input to have the size 200x66 and to remove parts outside of the track
    model.add(Cropping2D(cropping=((70, 24), (60, 60)), input_shape=input_shape))
    print(model.output_shape)

    # Preprocessing 2: Normalize images to be have pixel values between -0.5 and 0.5
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    print(model.output_shape)

    # 1st Layer: 5x5 Convolution and Max Pooling
    model.add(Convolution2D(24,5,5,activation='relu'))
    model.add(MaxPooling2D(border_mode='same'))
    print(model.output_shape)

    # 2nd Layer: 5x5 Convolution and Max Pooling
    model.add(Convolution2D(36,5,5,activation='relu'))
    model.add(MaxPooling2D(border_mode='same'))
    print(model.output_shape)

    # 3rd Layer: 5x5 Convolution and Max Pooling
    model.add(Convolution2D(48,5,5,activation='relu'))
    model.add(MaxPooling2D(border_mode='same'))
    print(model.output_shape)

    # 4th Layer: 3x3 Convolution
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    print(model.output_shape)

    # 5th Layer: 3x3 Convolution
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    print(model.output_shape)

    # Flatten
    model.add(Flatten())
    print(model.output_shape)

    # 6th - 9th Layer: Fully connected layers
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # Hyperparameters
    ratio_validation_set = 0.2
    batch_size = 64

    # Define loss and optimizer and train the model
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(X_train, y_train, batch_size=batch_size, validation_split=ratio_validation_set,
                               shuffle=True, nb_epoch=num_epochs, verbose=1)

    # Print the keys contained in the history object
    print(history_object.history.keys())

    # Plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    # Save model
    model.save(model_name)

# Main
X_train,y_train = load_data()
nvidia_net(X_train,y_train,10,'model.h5')
