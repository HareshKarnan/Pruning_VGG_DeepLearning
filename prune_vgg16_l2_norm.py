from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import np_utils
from keras.datasets import cifar10
from kerassurgeon import Surgeon
from kerassurgeon import Surgeon
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
import numpy as np
from keras.models import load_model

img_width, img_height = 32, 32
train_data_dir = "./data/train"
nb_train_samples = 4125
nb_validation_samples = 466
conv_layers = ['block1_conv1','block1_conv2','block2_conv1',
               'block2_conv2','block3_conv1','block3_conv2',
               'block3_conv3','block3_conv4','block4_conv1',
               'block4_conv2','block4_conv3','block4_conv4',
               'block5_conv1','block5_conv2','block5_conv3','block5_conv4',]
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2,
    horizontal_flip=True)

def train(model,epochs):
    bs = 32
    model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=['top_k_categorical_accuracy'])
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=bs,subset='training'),
                        steps_per_epoch=len(x_train)/bs, epochs=epochs,verbose=2)
    return model

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def takeSecond(elem):
    return elem[1]

if __name__== "__main__":

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    #load the VGG16 model
    model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

    #freeze VGG layers
    for layer in model.layers:
        layer.trainable = False
    last = model.output
    x = Flatten()(last)
    # x = Dense(1024, activation="relu")(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1024, activation="relu")(x)
    predictions = Dense(10, activation="softmax")(x)
    model = Model(input=model.input, output=predictions)
    print("starting initial training of the model")
    if os.path.isfile('./my_model_initial.h5'):
        print('model found on disk. Loading..')
        model = load_model('my_model_initial.h5')
        model.summary()
    else:
        print('model not found on disk.. fine tuning..')
        model = train(model,1)
        model.save('my_model_initial.h5')
        print('saved the trained model')

    # find the initial validation accuracy of the model before pruning
    acc = model.evaluate_generator(datagen.flow(x_train, y_train,subset='validation'),steps=10,verbose=0)[1]
    acc_pruned = acc
    # find the initial number of params before pruning
    initial_params = model.count_params()
    conv_index = 0 # index of which conv_layer the surgeon is working on. Start from 0th conv layer
    while(acc-acc_pruned<.05): # 5 percent loss is tolerable
        W = model.get_layer(conv_layers[conv_index]).get_weights()[0]
        # print(W.shape)
        ratio_list = []
        # fig, axs = plt.subplots(int(W.shape[3] ** 0.5), int(W.shape[3] ** 0.5), figsize=(20, 20))
        # fig.subplots_adjust(hspace=.5, wspace=.001)
        for i in range(W.shape[3]):
            filter_kernel_0 =  W[:, :, 0, i]
            filter_kernel_1 = W[:, :, 1, i]
            filter_kernel_2 = W[:, :, 2, i]
            l2_norm = np.linalg.norm(filter_kernel_0)+np.linalg.norm(filter_kernel_1)+np.linalg.norm(filter_kernel_1) #l2-norm of the kernel is computed
            ratio_list.append((i, l2_norm)) #append norm to ratio list along with channel number

        ratio_list = sorted(ratio_list, key=takeSecond, reverse=True)
        print(len(ratio_list))

        surgeon = Surgeon(model)
        channels_to_prune = [ratio_list[0][0],ratio_list[1][0],ratio_list[2][0]] # keep pruning 3 filters everytime
        surgeon.add_job('delete_channels', model.get_layer(conv_layers[conv_index]), channels=channels_to_prune)
        model = surgeon.operate()

        print('% of parameters now :: ',model.count_params()/initial_params)
        # train for 1 epochs
        print("Training the pruned model...")
        model=train(model,1)
        print("Trained the pruned model...")

        # find validation accuracy of the pruned model
        acc_pruned = model.evaluate_generator(datagen.flow(x_train, y_train,subset='validation'),steps=10,verbose=0)[1]
        print('accuracy of the pruned model :: ',acc_pruned)
        print('accuracy has dropped by :: ',acc-acc_pruned)

        if conv_index==len(conv_layers)-1:
            print("reached the last layer of pruning")
            break
        if(acc-acc_pruned<0.05):
            print('pruned all filters in layer :: '+str(conv_index)+'. Moving to the next layer to the right')
            conv_index=conv_index+1

        model.save('my_model_pruned_valacc_'+str(acc_pruned)+'.h5')
        print('saved the model ... ')



