# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")
import os
import random
from sklearn.model_selection import train_test_split
from PIL import Image

# Create Dataframe for Input and Output
data_dir = "PetImages"
input_path = []
label = []

for class_ in os.listdir("PetImages"):
    class_dir = os.path.join(data_dir, class_)
    if not os.path.isdir(class_dir): 
        continue
    for path in os.listdir(class_dir):
        input_path.append(os.path.join(class_dir, path))
        label.append(0 if class_ == "Cat" else 1)

df = pd.DataFrame()
df['images'] = input_path
df['label'] = label
df = df.sample(frac=1).reset_index(drop=True)
#df['label'] = df['label'].astype('str')
print(df.head())
 
plt.figure(figsize=(25,25))
temp = df[df['label'] == 1]['images']
start = random.randint(0, len(temp))
files = temp[start:start+25]

##### Obtain the images
#for index, file in enumerate(files):
    #plt.subplot(5, 5, index + 1)
    #img = load_img(file)
    #img = np.array(img)
   # plt.imshow(img)
  #  plt.title('Dogs')
 #   plt.axis('off')
#plt.show()
#####
# Removing defective files
def is_valid_image(path):
    try:
        with Image.open(path) as im:
            im.verify()     
        return True
    except Exception:
        return False

df['is_ok'] = df['images'].apply(lambda p: os.path.isfile(p) and os.path.getsize(p) > 0 and is_valid_image(p))
bad = df.loc[~df['is_ok'], 'images'].tolist()
df = df[df['is_ok']].drop(columns=['is_ok']).reset_index(drop=True)

df['label'] = df['label'].astype(str)
### Input split
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

### Create DataGenerators for the images
train_generator = ImageDataGenerator(
    rescale = 1./255, # Normalization of pixales values
    rotation_range = 20, #augmentations in order to avoid overfitting
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

val_generator = ImageDataGenerator(rescale = 1./255)


train_iterator = train_generator.flow_from_dataframe(
    train, 
    x_col = 'images',
    y_col = 'label',
    target_size= (128,128),
    batch_size= 64,
    class_mode = 'binary'
            )

val_iterator = val_generator.flow_from_dataframe(
    test, 
    x_col = 'images',
    y_col = 'label',
    target_size= (128,128),
    batch_size= 64,
    class_mode = 'binary',
    shuffle = False
            )

## Model creation
model = Sequential([
    Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128,(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(1, activation='sigmoid')
])

## Compile model
optimizer = Adam(learning_rate=3e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

ckpt = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

plateau = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

early = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_iterator, 
    epochs=15, 
    validation_data=val_iterator,
    callbacks=[ckpt, plateau, early]
    )

# Plot the resaults
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label="Training Accuracy")
plt.plot(epochs, val_acc, 'r', label="Validation Accuracy")
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()