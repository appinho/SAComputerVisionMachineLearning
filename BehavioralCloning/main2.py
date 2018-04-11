import dataset
import neutal_nets
import visualizer

# HYPERPARAMETERS
batch_size = 32
number_of_epochs = 5

train_samples,validation_samples = dataset.get_data()

# compile and train the model using the generator function
train_generator = dataset.generate(train_samples, batch_size)
validation_generator = dataset.generate(validation_samples, batch_size)

model = neutal_nets.le_net(0.5)

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
            samples_per_epoch= len(train_samples),
            validation_data=validation_generator,
            nb_val_samples=len(validation_samples),
            nb_epoch=number_of_epochs,
            verbose=1)
# save model
model.save('dg_model_lenet.h5')

print(history_object.history.keys())

visualizer.plot_loss(history_object)



