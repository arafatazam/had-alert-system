import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


processed_path = "hmdb51_processed"

actions = np.array(
    list(filter(lambda x: not x.startswith('.'), os.listdir(processed_path))))
label_map = {label: idx for idx, label in enumerate(actions)}


def create_model():
    """
    Create an LSTM deep learning model
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
              activation='relu', input_shape=(16, 34)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_data_and_labels():
    """
    Read the processed files and load them in the memory for training
    """
    data, labels = [], []
    for action in actions:
        action_dir = processed_path+'/'+action+'/'
        file_names = list(
            filter(lambda x: not x.startswith('.'), os.listdir(action_dir)))
        for file_name in file_names:
            loaded_data = np.load(os.path.join(
                processed_path, action, file_name))
            data.append(loaded_data)
            labels.append(label_map[action])
    data = np.array(data)
    labels = to_categorical(labels).astype(int)
    return data, labels


def run():
    model = create_model()
    X, y = get_data_and_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[early_stopping_callback])
    model.summary()

    res = model.evaluate(X_test, y_test)
    print('Test Accuracy:', res)

    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Task 3: Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.show()

    # Plot the training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Task 3: Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.show()

    model.save('model.h5')


if __name__ == '__main__':
    run()
