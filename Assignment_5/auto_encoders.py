from keras.layers import Input, Dense
from keras.models import Model


def create_auto_encoder(encoder_layers, decoder_layers):
    """
    Create an autoencoders with Keras, number of layers and neuron in each layer determines
    By 'encoder_layer' and 'decoder_layer'
    :param encoder_layers: a list that i'th element shows number of
            neurons in i'th layer of encoder
    :param decoder_layers: a list that i'th element shows number of
            neurons in i'th layer of encoder
    :return: autoencoder, encoder part and decoder part
    """

    # this is our input placeholder
    input_img = Input(shape=(encoder_layers[0],))

    for i_layer in range(1, len(encoder_layers)):
        if i_layer == 1:
            #
            encoded = Dense(encoder_layers[i_layer], activation='relu')(input_img)
        else:
            #
            encoded = Dense(encoder_layers[i_layer], activation='relu')(encoded)

    for i_layer in range(0, len(decoder_layers)):
        if i_layer == 0:

            if len(decoder_layers) == 1:
                #
                decoded = Dense(decoder_layers[i_layer], activation='sigmoid')(encoded)
            else:
                #
                decoded = Dense(decoder_layers[i_layer], activation='relu')(encoded)

        elif i_layer != len(decoder_layers)-1:
            #
            decoded = Dense(decoder_layers[i_layer], activation='relu')(decoded)
        else:
            #
            decoded = Dense(decoder_layers[i_layer], activation='sigmoid')(decoded)


    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # compile model
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['mse'])

    # return model
    return autoencoder, encoder
