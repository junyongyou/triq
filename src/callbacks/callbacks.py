import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from callbacks.csv_callback import MyCSVLogger


def create_callbacks(model_name, result_folder, other_callback=None, checkpoint=True, early_stop=True, metrics='accuracy'):
    """Creates callbacks for model training

    :param model_name: model name
    :param result_folder: folder to write to
    :param other_callback: other evaluation callbacks
    :param checkpoint: flag to use checkpoint or not
    :param early_stop: flag to use early_stop or not
    :param metrics: evaluation metrics for writing to checkpoint file
    :return: callbacks
    """

    callbacks = []
    if other_callback is not None:
        callbacks.append(other_callback)
    csv_log_file = os.path.join(result_folder, model_name + '.log')
    csv_logger = MyCSVLogger(csv_log_file, model_name, append=True, separator=';')
    callbacks.append(csv_logger)
    if early_stop:
        callbacks.append(EarlyStopping(monitor='plcc', min_delta=0.001, patience=40, mode='max'))
    if checkpoint:
        if metrics == None:
            mcp_file = os.path.join(result_folder, '{epoch:01d}_{loss:.4f}_{val_loss:.4f}.h5')
        else:
            if metrics == 'accuracy':
                mcp_file = os.path.join(result_folder, '{epoch:01d}_{loss:.4f}_{accuracy:.4f}_{val_loss:.4f}_{val_accuracy:.4f}.h5')
            elif metrics == 'mae':
                mcp_file = os.path.join(result_folder, '{epoch:01d}_{loss:.4f}_{mae:.4f}_{val_loss:.4f}_{val_mae:.4f}.h5')
            elif metrics == 'categorical_crossentropy':
                mcp_file = os.path.join(result_folder, '{epoch:01d}_{loss:.4f}_{categorical_crossentropy:.4f}_{val_loss:.4f}_{val_categorical_crossentropy:.4f}.h5')
            elif metrics == 'distribution_loss':
                mcp_file = os.path.join(result_folder, '{epoch:01d}_{loss:.4f}_{distribution_loss:.4f}_{val_loss:.4f}_{val_distribution_loss:.4f}.h5')
            else:
                mcp_file = os.path.join(result_folder, '{epoch:01d}_{loss:.4f}_{val_loss:.4f}.h5')
        mcp = ModelCheckpoint(mcp_file, save_best_only=True, save_weights_only=True, monitor='plcc', verbose=1, mode='max')
        callbacks.append(mcp)

    # tensorboard_callback = TensorBoard(log_dir=result_folder, histogram_freq=1)
    # callbacks.append(tensorboard_callback)

    return callbacks
