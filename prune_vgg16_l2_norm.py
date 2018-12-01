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
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
import numpy as np

img_width, img_height = 32, 32
train_data_dir = "./data/train"
nb_train_samples = 4125
nb_validation_samples = 466

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
    train(model,1)
    model.save('my_model_initial.h5')
    #model.summary()

    acc = model.evaluate_generator(datagen.flow(x_train, y_train,subset='validation'),steps=10,verbose=0)[1]

    acc_pruned = acc
    initial_params = model.count_params()
    while(acc-acc_pruned<.05): # 5 percent loss is tolerable
        W = model.get_layer('block1_conv1').get_weights()[0]
        # print(W.shape)
        ratio_list = []
        # fig, axs = plt.subplots(int(W.shape[3] ** 0.5), int(W.shape[3] ** 0.5), figsize=(20, 20))
        # fig.subplots_adjust(hspace=.5, wspace=.001)
        for i in range(W.shape[3]):
            for ch in range(3): # different channels
                filter_wt = (np.pad(W[:, :, ch, i], 5, pad_with))
                fft_computed = np.abs(np.fft.fft2(filter_wt))
                fft_shifted = np.fft.fftshift(fft_computed)
                # center_val = int(fft_shifted.shape[0] // 2)
                # breadth = 3
                # dc = np.sum(fft_shifted[center_val - breadth:center_val + breadth, center_val - breadth:center_val + breadth])
                # ac = np.sum(fft_shifted) - dc
                l2_norm = np.linalg.norm(fft_shifted)
                ratio_list.append((i, l2_norm))

        ratio_list = sorted(ratio_list, key=takeSecond, reverse=True)

        surgeon = Surgeon(model)
        channel_to_prune = [ratio_list[0][0],ratio_list[1][0],ratio_list[2][0]] # keep pruning 3 filters everytime
        surgeon.add_job('delete_channels', model.get_layer('block1_conv1'), channels=channel_to_prune)
        model = surgeon.operate()
        print('% of parameters now :: ',model.count_params()/initial_params)
        # train for 1 epochs
        print("Training the pruned model...")
        train(model,1)

        print("Trained the pruned model...")
        # find the validation accuracy of the pruned model
        acc_pruned = model.evaluate_generator(datagen.flow(x_train, y_train,subset='validation'),steps=10,verbose=0)[1]
        print('accuracy of the pruned model :: ',acc_pruned)
        print('accuracy has dropped by :: ',acc-acc_pruned)
        model.save('my_model_pruned.h5')
        print('saved the model ... ')



