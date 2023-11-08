from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras


base_model = keras.applications.ResNet50(weights='imagenet',
                                         input_shape=(224, 224, 3),
                                         include_top=False)
# base_model.summary()
base_model.trainable = False
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAvgPool2D()(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

datagen_train = ImageDataGenerator(
    samplewise_center=True,  # set each sample mean to 0
    rotation_range=10,
    # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,
    # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,
)  # we don't expect Bo to be upside-down so we will not flip vertically
datagen_valid = ImageDataGenerator(samplewise_center=True)
train_iter = datagen_train.flow_from_directory('C:/data/taco_and_burrito/train',
                                               target_size=(224, 224),
                                               color_mode='rgb',
                                               class_mode='binary',
                                               batch_size=8)
valid_iter = datagen_valid.flow_from_directory('C:/data/taco_and_burrito/test',
                                               target_size=(224, 224),
                                               color_mode='rgb',
                                               class_mode='binary',
                                               batch_size=8)

model.fit(train_iter, steps_per_epoch=12, validation_data=valid_iter,
          validation_steps=4, epochs=20)

