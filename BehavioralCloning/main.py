import data
import net

X_train,y_train = data.load_data()
data.draw_data(y_train)
net.one_layer_net(X_train,y_train,3,'model_1.h5')
net.one_layer_net_with_preprocessing(X_train,y_train,1,'model_1_pre.h5')
net.le_net(X_train,y_train,3,'model_lenet.h5',0.5)
net.alex_net(X_train,y_train,9,'model_alexnet.h5')
net.nvidia_net(X_train,y_train,10,'model_nvidia.h5')
