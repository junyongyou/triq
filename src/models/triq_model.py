"""
Main function to build TRIQ.
"""
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from backbone.resnet50 import ResNet50
from backbone.vgg16 import VGG16
from models.transformer_iqa import TriQImageQualityTransformer
import tensorflow as tf


def create_triq_model(n_quality_levels,
                      input_shape=(None, None, 3),
                      backbone='resnet50',
                      transformer_params=(2, 32, 8, 64),
                      maximum_position_encoding=193,
                      vis=False):
    """
    Creates the hybrid TRIQ model
    :param n_quality_levels: number of quality levels, use 5 to predict quality distribution
    :param input_shape: input shape
    :param backbone: bakbone nets, supports ResNet50 and VGG16 now
    :param transformer_params: Transformer parameters
    :param maximum_position_encoding: the maximal number of positional embeddings
    :param vis: flag to visualize attention weight maps
    :return: TRIQ model
    """
    inputs = Input(shape=input_shape)
    if backbone == 'resnet50':
        backbone_model = ResNet50(inputs,
                                  return_feature_maps=False, return_last_map=True)
    elif backbone == 'vgg16':
        backbone_model = VGG16(inputs, return_last_map=True)
    else:
        raise NotImplementedError

    C5 = backbone_model.output

    dropout_rate = 0.1

    transformer = TriQImageQualityTransformer(
        num_layers=transformer_params[0],
        d_model=transformer_params[1],
        num_heads=transformer_params[2],
        mlp_dim=transformer_params[3],
        dropout=dropout_rate,
        n_quality_levels=n_quality_levels,
        maximum_position_encoding=maximum_position_encoding,
        vis=vis
    )
    outputs = transformer(C5)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    input_shape = [None, None, 3]
    # input_shape = [768, 1024, 3]
    # input_shape = [500, 500, 3]
    # input_shape = [384, 512, 3]
    # model = cnn_transformer(n_quality_levels=5, input_shape=input_shape, backbone='vgg16')
    model = create_triq_model(n_quality_levels=5, input_shape=input_shape, backbone='resnet50')
