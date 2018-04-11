from keras.models import Sequential
from keras.layers import Dropout,Flatten,Dense,Lambda,Convolution2D,MaxPooling2D, Cropping2D
import matplotlib.pyplot as plt

# Hyperparameters
ratio_validation_set = 0.2
batch_size = 64

def one_layer_net(X_train,y_train,num_epochs,model_name):
    # define network
    input_shape = X_train.shape[1:4]
    print(input_shape)
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1))

    # training
    train_model(model,X_train,y_train,num_epochs,model_name)

def one_layer_net_with_preprocessing(X_train,y_train,num_epochs,model_name):
    # define network
    input_shape = X_train.shape[1:4]
    print(input_shape)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1))

    # training
    train_model(model,X_train,y_train,num_epochs,model_name)

def le_net(X_train,y_train,num_epochs,model_name, p_dropout):
    input_shape = X_train.shape[1:4]
    print(input_shape)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    print(model.output_shape)
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(6,5,5,activation='relu'))
    print(model.output_shape)
    model.add(MaxPooling2D())
    if p_dropout > 0:
        model.add(Dropout(p_dropout))
    print(model.output_shape)
    model.add(Convolution2D(6,5,5,activation='relu'))
    print(model.output_shape)
    model.add(MaxPooling2D())
    if p_dropout > 0:
        model.add(Dropout(p_dropout))
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(1200,activation='relu'))
    model.add(Dense(84,activation='relu'))
    model.add(Dense(1))

    train_model(model,X_train,y_train,num_epochs,model_name)

def alex_net(X_train,y_train,num_epochs,model_name):
    input_shape = X_train.shape[1:4]
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    print(model.output_shape)
    model.add(Cropping2D(cropping=((65, 25), (0, 0))))
    print(model.output_shape)
    # Layer 1 - Convolutional
    model.add(Convolution2D(16,5,5,subsample=(2, 2), activation='relu'))
    print(model.output_shape)
    model.add(MaxPooling2D(border_mode='same'))
    print(model.output_shape)
    # Layer 2 - Convolutional
    model.add(Convolution2D(32,5, 5, subsample=(2, 2), activation='relu'))
    print(model.output_shape)
    model.add(MaxPooling2D(border_mode='same'))
    print(model.output_shape)
    model.add(Dropout(0.2))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    # Layer 3 - Convolutional
    model.add(Convolution2D(64,3, 3, subsample=(2, 2), activation='relu'))
    print(model.output_shape)
    model.add(Dropout(0.5))

    # Layer 6 - Flatten
    model.add(Flatten())
    print(model.output_shape)

    # Layer 7 - Fully Connected
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))

    # Layer 8 - Fully Connected
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.5))

    # Layer 9 - Fully Connected
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(0.5))

    # Layer 10 - Fully Connected
    model.add(Dense(1))

    train_model(model,X_train,y_train,num_epochs,model_name)

def nvidia_net(X_train,y_train,num_epochs,model_name):
    input_shape = X_train.shape[1:4]
    print(input_shape)
    model = Sequential()
    # Cropping images
    model.add(Cropping2D(cropping=((70, 26), (60, 60)), input_shape=input_shape))
    print(model.output_shape)

    # Normalization
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    # 1st 5x5 Convolution
    model.add(Convolution2D(24,5,5,activation='relu'))
    model.add(MaxPooling2D(border_mode='same'))
    print(model.output_shape)

    # 2nd 5x5 Convolution
    model.add(Convolution2D(36,5,5,activation='relu'))
    model.add(MaxPooling2D(border_mode='same'))
    print(model.output_shape)

    # 3rd 5x5 Convolution
    model.add(Convolution2D(48,5,5,activation='relu'))
    model.add(MaxPooling2D(border_mode='same'))
    print(model.output_shape)

    # 4th 3x3 Convolution
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    print(model.output_shape)

    # 5th 3x3 Convolution
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    print(model.output_shape)

    # Flatten
    model.add(Flatten())
    print(model.output_shape)

    # Fully connected layers
    #model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    train_model(model,X_train,y_train,num_epochs,model_name)


def train_model(model,X_train,y_train,num_epochs,model_name):
    # define loss and optimizer and train the model
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(X_train,y_train,batch_size=batch_size,validation_split=ratio_validation_set,
              shuffle=True,nb_epoch=num_epochs,verbose=1)

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    # save model
    model.save(model_name)