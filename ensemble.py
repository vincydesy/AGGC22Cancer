import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.layers import Dense, Flatten, Average
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight


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


def create_base_model(model_class, model_name, input_shape=(224, 224, 3), num_classes=5):
    """Crea un modello base utilizzando una particolare architettura ResNet con nomi unici per i layer."""
    base_model = model_class(weights='imagenet', include_top=False, input_shape=input_shape)

    # Rinomina i layer per garantire che abbiano nomi unici
    for layer in base_model.layers:
        layer._name = f"{model_name}_{layer.name}"

    x = Flatten(name=f"{model_name}_flatten")(base_model.output)
    x = Dense(1024, activation='relu', name=f"{model_name}_dense_1")(x)
    output = Dense(num_classes, activation='softmax', name=f"{model_name}_output")(x)

    model = Model(inputs=base_model.input, outputs=output, name=model_name)
    return model


def create_ensemble_model(input_shape=(224, 224, 3), num_classes=5):
    """Crea un ensemble di modelli ResNet50, ResNet101, ResNet152."""
    # Crea i tre modelli di base con nomi unici
    model_50 = create_base_model(ResNet50, "ResNet50", input_shape, num_classes)
    model_101 = create_base_model(ResNet101, "ResNet101", input_shape, num_classes)
    model_152 = create_base_model(ResNet152, "ResNet152", input_shape, num_classes)

    # Media dei layer di output dei modelli
    ensemble_output = Average(name="ensemble_average")([model_50.output, model_101.output, model_152.output])

    # Crea il modello ensemble
    ensemble_model = Model(inputs=[model_50.input, model_101.input, model_152.input], outputs=ensemble_output,
                           name="EnsembleModel")

    return ensemble_model


def train_ensemble_model(image_dir):
    """Addestra il modello ensemble sui dati forniti in image_dir."""

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

    # Crea il modello ensemble
    ensemble_model = create_ensemble_model(input_shape=(224, 224, 3), num_classes=num_classes)

    # Compila il modello
    ensemble_model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    # Addestra il modello ensemble
    history = ensemble_model.fit(
        [train_generator, train_generator, train_generator],  # Triplica il generatore per l'ensemble
        validation_data=([validation_generator, validation_generator, validation_generator]),
        epochs=10,
        class_weight=class_weights_dict,
        verbose=1
    )

    # Salva il modello ensemble addestrato
    criterion_name = os.path.basename(image_dir)
    model_filename = f'ensembleResNet_{criterion_name}.h5'
    ensemble_model.save(model_filename)

    return ensemble_model


# Esempio di utilizzo
trained_model = train_ensemble_model('/media/vincydesy/HData_2/Istologico/script elaborazione patch matlab/esperimento 3/criterio_1')
