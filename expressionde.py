import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, MaxPool2D, AveragePooling2D
from keras.layers import Conv2D, Activation, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# cpu configuration
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
sess = tf.Session(config=config)
keras.backend.set_session(sess)
#

num_classes = 7 # sad, happy, surprise, angry, fear, nutral, disgust
batch_size = 256
epoch = 2

with open("fer2013.csv") as f:
    content = f.readlines()
lines = np.array(content)
num_instance =lines.size
# train test
x_train, y_train, x_test, y_test = [], [], [], []

for i in range(1,num_instance):
    try:
        emotion, img, usage = lines[i].split(",")
        val = img.split(" ")
        pixels = np.array(val, 'float32')
        emotion = keras.utils.to_categorical(emotion, num_classes)
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
        print("", end="")

# data transformation for train and test sets
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255 #normalize inputs between [0, 1]
x_test /= 255
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# creating the sequential model
imodel_fe = Sequential()

#1st convolution layer
imodel_fe.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
imodel_fe.add(MaxPool2D(pool_size=(5, 5), strides=(2, 2)))

#2nd c layer
imodel_fe.add(Conv2D(64, (3, 3), activation='relu'))#128
imodel_fe.add(Conv2D(64, (3,3), activation='relu'))
imodel_fe.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

#3rd c layer
imodel_fe.add(Conv2D(64, (3, 3), activation='relu'))#128
imodel_fe.add(Conv2D(64, (3,3), activation='relu'))
imodel_fe.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

imodel_fe.add(Flatten())
# fully connected neural network
imodel_fe.add(Dense(1024, activation='relu'))
imodel_fe.add(Dropout(0.2))
imodel_fe.add(Dense(1024, activation='relu'))
imodel_fe.add(Dropout(0.2))

imodel_fe.add(Dense(num_classes, activation='softmax'))

#--------------------------------------
#batch process
gen = ImageDataGenerator()# this is used to normalise images in format keras can understand
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
imodel_fe.compile(loss='categorical_crossentropy'
    , optimizer=keras.optimizers.Adam()
    , metrics=['accuracy']
)
#---------------------------------------------

fit = True
if fit == True:
	#model.fit_generator(x_train, y_train, epochs=epochs) #train for all trainset
	imodel_fe.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epoch) #train for randomly selected one
else:
	imodel_fe.load_weights('model33.h5') #load weights


# ------------------------------
# function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')

    plt.show()
# ------------------------------
imodel_fe.save('model2.h5')
#--------------------------------
monitor_testset_results = True
if monitor_testset_results == True:
    # make predictions for test set
    predictions = imodel_fe.predict(x_test)

    index = 0
    for i in predictions:
        if index < 30 and index >= 20:
            # print(i) #predicted scores
            # print(y_test[index]) #actual scores

            testing_img = np.array(x_test[index], 'float32')
            testing_img = testing_img.reshape([48, 48]);

            plt.gray()
            plt.imshow(testing_img)
            plt.show()

            print(i)

            emotion_analysis(i)
            print("----------------------------------------------")
        index = index + 1

#------------------------------------------------------------------------------
#











