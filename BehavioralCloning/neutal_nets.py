from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,MaxPooling2D, Cropping2D, Dropout

ch, row, col = 3, 160, 320  # Trimmed image format

def le_net(p_dropout):
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    print(model.output_shape)
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    print(model.output_shape)
    model.add(MaxPooling2D())
    if p_dropout > 0:
        model.add(Dropout(p_dropout))
    print(model.output_shape)
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    print(model.output_shape)
    model.add(MaxPooling2D())
    if p_dropout > 0:
        model.add(Dropout(p_dropout))
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(1200, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))
    return model