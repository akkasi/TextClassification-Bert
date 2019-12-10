from utils import *
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


batch_size = 32
Epoach = 10
max_seq_length = 50

tokenizer = create_tokenizer_from_hub_module(bert_path)
Train,Label = DataPreparation()
train_examples = convert_text_to_examples(Train, Label)

(train_input_ids, train_input_masks, train_segment_ids, train_labels) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)


# Build model
def build_model(max_seq_length):
    in_id = Input(shape=(max_seq_length,), name="input_ids")
    in_mask = Input(shape=(max_seq_length,), name="input_masks")
    in_segment = Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]
    bert_output = BertLayer(n_fine_tune_layers=1, pooling="first")(bert_inputs)
    bert_output = tf.reshape(bert_output, [-1, 768, 1])

    # CNN ##################################################################
    #     cnn = Conv1D(128,5,input_shape=(None, 768),activation='relu')(bert_output)
    #     cnn = Conv1D(128,5,input_shape=(None, 768),activation='relu')(cnn)
    #     cnn = Flatten()(cnn)
    #     pred = tf.keras.layers.Dense(1, activation='sigmoid')(cnn)

    # DENSE
    dense = Dense(256, activation='sigmoid')(bert_output)
    dense = Flatten()(dense)
    pred = Dense(1, activation='sigmoid')(dense)

    # #LSTM
    #     lstm = LSTM(128,activation='relu',return_sequences=True)(bert_output)
    #     lstm = LSTM(128,activation='relu',return_sequences=False)(lstm)
    #     lstm = Flatten()(lstm)
    #     pred = tf.keras.layers.Dense(1,activation='sigmoid')(lstm)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

model = build_model(max_seq_length)
initialize_vars(sess)
model.fit(
    [train_input_ids, train_input_masks, train_segment_ids],
    train_labels,
    epochs=1,
    batch_size= batch_size
)

model.evaluate([train_input_ids, train_input_masks, train_segment_ids], train_labels)