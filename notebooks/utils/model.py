import tensorflow as tf


def example_encoder_decoder_model(n_features: int, n_features_pred: int, n_static: int, n_past: int, n_future: int):
    encoder_inputs = tf.keras.layers.Input(batch_shape=(1, n_past, n_features))
    encoder_l1 = tf.keras.layers.LSTM(2048, name="Enc1", recurrent_activation="sigmoid", dropout=.4, recurrent_dropout=0, stateful=True,
                                      return_sequences=True, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]
    encoder_l2 = tf.keras.layers.LSTM(2048, name="Enc2", recurrent_activation="sigmoid", dropout=.4, recurrent_dropout=0, stateful=True,
                                      return_state=True)
    encoder_outputs2 = encoder_l2(encoder_outputs1[0])
    encoder_states2 = encoder_outputs2[1:]

    static_inputs = tf.keras.layers.Input(shape=(n_static,), name="Static")
    static_outputs = tf.keras.layers.Dense(256, name="D1", activation='tanh')(static_inputs)

    concat_outputs = tf.keras.layers.concatenate([encoder_outputs2[0], static_outputs], name="Concat")

    decoder_inputs = tf.keras.layers.RepeatVector(n_future)(concat_outputs)

    decoder_l1 = tf.keras.layers.LSTM(2048, name="Dec1", recurrent_activation="sigmoid", dropout=.4, recurrent_dropout=0, stateful=True,
                                      return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
    decoder_l2 = tf.keras.layers.LSTM(2048, name="Dec2", recurrent_activation="sigmoid", dropout=.4, recurrent_dropout=0, stateful=True,
                                      return_sequences=True)(decoder_l1, initial_state=encoder_states2)
    decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features_pred))(decoder_l2)

    model = tf.keras.models.Model([encoder_inputs, static_inputs], decoder_outputs2)

    model.summary()

    return model
