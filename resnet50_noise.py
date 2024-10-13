import os
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

def zero_one_loss(y_true, y_pred):
    """Funzione che implementa la 0-1 loss come metrica."""
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_true_labels = K.argmax(y_true, axis=-1)
    return K.cast(K.not_equal(y_true_labels, y_pred_labels), K.floatx())
def extract_label_from_filename(filename):
    """Funzione per estrarre l'etichetta dal nome dell'immagine."""
    parts = filename.split('_')
    label = parts[-1].replace('.tiff', '')
    return label


def create_dataframe(image_dir):
    """Crea un dataframe che mappa i percorsi delle immagini alle loro etichette."""
    file_paths = []
    labels = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.tiff'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                labels.append(extract_label_from_filename(file))

    df = pd.DataFrame({
        'filename': file_paths,
        'class': labels
    })
    return df


def train_resnet_on_criterion50(image_dir):
    """Funzione per addestrare una ResNet sui dati forniti in image_dir."""

    # Crea il dataframe dalle immagini e dalle etichette
    df = create_dataframe(image_dir)

    # Codifica le etichette come valori categoriali
    le = LabelEncoder()
    df['class_encoded'] = le.fit_transform(df['class'])
    num_classes = len(le.classes_)

    # Calcola i pesi per ciascuna classe
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=df['class_encoded']
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Configura ImageDataGenerator per il caricamento in batch
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    # Generatore per il training
    train_generator = datagen.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    # Generatore per la validation
    validation_generator = datagen.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    print(f"Number of classes: {num_classes}")

    # Carica la ResNet50 pre-addestrata senza i layer superiori (top)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Congela i pesi della rete pre-addestrata
    for layer in base_model.layers:
        layer.trainable = True

    # Aggiungi i nuovi layer alla rete
    x = Flatten()(base_model.output)
    x = Dense(1024, activation='relu')(x)

    # Aggiungi il Noise Layer
    noise = Dense(num_classes, activation='linear',
                  kernel_initializer='identity',
                  kernel_regularizer=l2(1e-4))(x)

    # Somma il Noise Layer ai logit finali
    output = Dense(num_classes, activation='softmax')(noise)

    # Crea il nuovo modello
    model = Model(inputs=base_model.input, outputs=output)

    # Compila il modello con Huber Loss
    model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9),
                  loss=Huber(delta=1.0),
                  metrics=['accuracy', zero_one_loss])

    # Addestra il modello
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        class_weight=class_weights_dict,
        verbose=1
    )

    # Rimuovi il Noise Layer per il testing
    #output_without_noise = Dense(num_classes, activation='softmax')(x)
   # model_without_noise = Model(inputs=base_model.input, outputs=output_without_noise)

    # Compila il modello senza il Noise Layer
   # model_without_noise.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9),
    #                            loss=Huber(delta=1.0),  # Usare la Huber Loss
     #                           metrics=['accuracy'])

    # Salva il modello addestrato
    criterion_name = os.path.basename(image_dir)
    model_filename = f'trainedResNetnoise_{criterion_name}.h5'
    model.save(model_filename)

    return model


# Esempio di chiamata alla funzione
train_resnet_on_criterion50('/media/vincydesy/HData_2/Istologico/script elaborazione patch matlab/esperimento 3/criterio_1')
# train_resnet_on_criterion50('/kaggle/input/esperimento-3/criterio_2')
# train_resnet_on_criterion50('/kaggle/input/esperimento-3/criterio_3')
