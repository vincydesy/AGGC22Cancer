import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Seleziona GPU 0 e GPU 1
        tf.config.experimental.set_visible_devices([gpus[1], gpus[2]], 'GPU')

        # Opzionale: attiva la crescita dinamica della memoria per entrambe le GPU
        tf.config.experimental.set_memory_growth(gpus[1], True)
        tf.config.experimental.set_memory_growth(gpus[2], True)
    except RuntimeError as e:
        print(e)


def save_confusion_matrix(conf_mat, class_labels, save_path, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    file_path = os.path.join(save_path, f'{model_name}.png')

    # Salva la matrice di confusione nel percorso specificato
    plt.savefig(file_path)
    plt.close()  # Chiude la figura per liberare memoria

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


def test_resnet_on_test_set(model_filename, test_image_dir):
    """Funzione per testare il modello ResNet su un set di immagini di test."""

    save_path = os.path.dirname(model_filename)
    model_name = os.path.splitext(os.path.basename(model_filename))[0]
    # Caricare il modello addestrato
    model = tf.keras.models.load_model(model_filename, custom_objects={'zero_one_loss': zero_one_loss})
    #model = tf.keras.models.load_model(model_filename)

    # Creare il dataframe per il set di test
    test_df = create_dataframe(test_image_dir)

    # Configura ImageDataGenerator per il caricamento in batch
    datagen = ImageDataGenerator(rescale=1. / 255)

    # Generatore per il test
    test_generator = datagen.flow_from_dataframe(
        test_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Eseguire le predizioni sulle immagini di test
    predictions = model.predict(test_generator, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)

    # Etichette reali
    actual_labels = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Calcolare la matrice di confusione
    conf_mat = confusion_matrix(actual_labels, predicted_labels)

    # Calcolare le metriche di valutazione
    accuracy = np.sum(predicted_labels == actual_labels) / len(actual_labels)
    precision = precision_score(actual_labels, predicted_labels, average='macro')
    recall = recall_score(actual_labels, predicted_labels, average='macro')
    f1 = f1_score(actual_labels, predicted_labels, average='macro')

    # Specificità e Sensibilità per classe
    sensitivity = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    specificity = [(np.sum(conf_mat) - np.sum(conf_mat[i, :]) - np.sum(conf_mat[:, i]) + conf_mat[i, i]) / (
            np.sum(conf_mat) - np.sum(conf_mat[i, :])) for i in range(len(class_labels))]
    support = np.sum(conf_mat, axis=1)

    # Calcolo AUC per ogni classe con l'approccio one-vs-all
    actual_labels = np.array(actual_labels)
    auc = []
    for i in range(len(class_labels)):
        actual_binary = (actual_labels == i).astype(int)
        auc_value = roc_auc_score(actual_binary, predictions[:, i])
        auc.append(auc_value)

    # Visualizzare i risultati
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Specificity: {np.mean(specificity):.2f}')
    print(f'Sensitivity: {np.mean(sensitivity):.2f}')
    print(f'Mean AUC: {np.mean(auc):.2f}')

    save_confusion_matrix(conf_mat, class_labels, save_path, model_name)

    # Visualizzare metriche per classe
    for i in range(len(class_labels)):
        print(
            f'Class {class_labels[i]}: Sensitivity {sensitivity[i]:.2f}, Specificity {specificity[i]:.2f}, Support {support[i]}')


# Esempio di richiamo della funzione
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
test_resnet_on_test_set('/home/vincydesy/AGGC22/trainedResNetnoise_criterio_1.h5',
                        '/media/vincydesy/HData_2/AGGC22_Mio/test_set_patches')
