import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Embedding, Dense, BatchNormalization, PReLU, Dropout, Input
from tensorflow.python.keras.regularizers import l2
from src.layer.AttentionLayer import AttentionLayer, Dice

class DIN(Model):
    def __init__(self,feature_columns, behavior_feature_list, attention_hidden_units=(80, 40),
                 ffn_hidden_units=(80, 40), attention_activation='prelu', ffn_activation='prelu',
                 max_len=40, dnn_dropout=0.,embed_reg=1e-4):
        '''

        :param feature_columns:
        :param behavior_feature_list:
        :param attention_hidden_units:
        :param ffn_hidden_units:
        :param attention_activation:
        :param ffn_activation:
        :param max_len:
        :param dnn_droupout:
        :param embed_reg:
        '''
        super().__init__()
        self.max_len = max_len
        self.dense_feature_column, self.sparse_feature_column = feature_columns

        # len
        self.other_sparse_len = len(self.sparse_feature_column) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_column)
        self.behavior_num = len(behavior_feature_list)

        # other embedding layers
        self.sparse_feature_embedding = [
            Embedding(
                input_dim=feat['feat_dim'],
                input_length=1,
                output_dim=feat['embed_dim'],
                embeddings_initializer='random_uniform',
                embeddings_regularizer=l2(embed_reg)
            )
            for index, feat in enumerate(self.sparse_feature_column)
            if feat['feat'] not in behavior_feature_list
        ]

        self.behavior_seq_embedding = [
            Embedding(
                input_dim=feat['feat_dim'],
                input_length=1,
                output_dim=feat['embed_dim'],
                embeddings_initializer='random_uniform',
                embeddings_regularizer=l2(embed_reg)
            )
            for index, feat in enumerate(self.sparse_feature_column)
            if feat['feat'] in behavior_feature_list
        ]

        # Attention Layer
        self.attention_layer = AttentionLayer(activation=attention_activation, att_hidden_units=attention_hidden_units)

        self.bn = BatchNormalization(trainable=True)
        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation == 'prelu' else Dice()) for unit in ffn_hidden_units]

        self.dropout = Dropout(dnn_dropout)
        self.output_layer = Dense(1)

    def call(self, inputs):
        # dense_inputs and sparse_inputs is empty
        # seq_inputs (None, maxlen, behavior_num)
        # item_inputs (None, behavior_num)
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs
        # attention ---> mask, if the element of seq_inputs is equal 0, it must be filled in.
        mask = tf.cast(tf.not_equal(seq_inputs[:, :, 0], 0), dtype=tf.float32)  # (None, maxlen)
        # other
        other_info = dense_inputs
        for i in range(self.other_sparse_len):
            other_info = tf.concat([other_info, self.sparse_feature_embedding[i](sparse_inputs[:, i])], axis=-1)

        # seq, item embedding and category embedding should concatenate
        seq_embed = tf.concat([self.behavior_seq_embedding[i](seq_inputs[:, :, i]) for i in range(self.behavior_num)],
                              axis=-1)
        item_embed = tf.concat([self.behavior_seq_embedding[i](item_inputs[:, i]) for i in range(self.behavior_num)], axis=-1)

        # att
        user_info = self.attention_layer([item_embed, seq_embed, seq_embed, mask])  # (None, d * 2)

        # concat user_info(att hist), cadidate item embedding, other features
        if self.dense_len > 0 or self.other_sparse_len > 0:
            info_all = tf.concat([user_info, item_embed, other_info], axis=-1)
        else:
            info_all = tf.concat([user_info, item_embed], axis=-1)

        info_all = self.bn(info_all)

        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)

        info_all = self.dropout(info_all)
        outputs = tf.nn.sigmoid(self.output_layer(info_all))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(self.dense_len,), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_len,), dtype=tf.int32)
        seq_inputs = Input(shape=(self.max_len, self.behavior_num), dtype=tf.int32)
        item_inputs = Input(shape=(self.behavior_num,), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs, item_inputs],
                       outputs=self.call([dense_inputs, sparse_inputs, seq_inputs, item_inputs])).summary()

def test_model():
    dense_features = [{'feat': 'a'}, {'feat': 'b'}]
    sparse_features = [{'feat': 'item_id', 'feat_dim': 100, 'embed_dim': 8},
                       {'feat': 'cate_id', 'feat_dim': 100, 'embed_dim': 8},
                       {'feat': 'adv_id', 'feat_dim': 100, 'embed_dim': 8}]
    behavior_list = ['item_id', 'cate_id']
    features = [dense_features, sparse_features]
    model = DIN(features, behavior_list)
    model.summary()