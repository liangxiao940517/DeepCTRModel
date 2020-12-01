import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Dense,Embedding,Concatenate,Dropout,Input,Layer
from tensorflow.keras.regularizers import l2
from conf.CommonConfig import CommonConfig as cf
from conf.ModelConfig import ModelConfig as mf


class WideDeep(tf.keras.Model):
    def __init__(self, feature_columns, hidden_units, activation='relu', dnn_droupout='0', embed_reg=1e-4):
        '''
        Wide&Deep Model
        :param feature_columns:
        :param hidden_units:
        :param activation:
        :param dnn_droupout:
        :param embed_reg:
        '''
        super(WideDeep,self).__init__()
        self.dense_feature_column, self.sparse_feature_column = feature_columns
        self.sparse_feature_embedding = {
            'sparse_embedding_' + str(i) : Embedding(input_dim=feat['feat_dim'],
                                          input_length=1,
                                          output_dim=feat['embed_dim'],
                                          embeddings_initializer='random_uniform',
                                          embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_column)
        }
        self.deep_network = [
            Dense(units=unit,activation=activation)
            for unit in hidden_units
        ]
        self.dropout = Dropout(mf.DROUPOUT_RATIO)
        self.deep_output = Dense(units=1,activation='linear')
        self.wide_layer = Dense(units=1, activation=None)
        self.output_layer = Dense(units=1,activation='sigmoid')



    def call(self, inputs, training=None, mask=None):
        dense_input, sparse_input = inputs
        sparse_embed = tf.concat([
            self.sparse_feature_embedding['sparse_embedding_{}'.format(index)](sparse_input[:, index])
            for index in range(sparse_input.shape[1])
        ], axis=-1)

        # deep part
        deep_x = tf.concat([dense_input, sparse_embed], axis=-1)
        for hidden_layer in self.deep_network:
            deep_x = hidden_layer(deep_x)
            if mf.DROUPOUT_OR_NOT:
                deep_x = self.dropout(deep_x)
        deep_output = self.deep_output(deep_x)

        # wide part
        wide_output = self.wide_layer(sparse_embed)

        output = self.output_layer(mf.DEEP_OUTPUT_RATIO * deep_output + mf.WIDE_OUTPUT_RATIO * wide_output)

        return output
