import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Embedding, Dense
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from conf.CommonConfig import CommonConfig as cf
from conf.ModelConfig import ModelConfig as mf


class MLP(keras.Model):
    def __init__(self,feature_columns,hidden_units,activation='relu',
                 droupout=0.,embed_reg=1e-4):
        '''
        MLP
        :param feature_columns: feature column list: dense_feature_column + sparse_feature_column
        :param hidden_units: MLP network
        :param activation: activation fucntion
        :param droupout: droupout rate
        :param embed_reg: The regularizer of embedding
        '''
        super(MLP, self).__init__()
        self.dense_feature_column, self.sparse_feature_column = feature_columns
        self.sparse_feature_embedding = {
            "sparse_embedding_" + str(index) : Embedding(
                input_dim=feat['feat_dim'],
                input_length=1,
                output_dim=feat['embed_dim'],
                embeddings_initializer='random_uniform',
                embeddings_regularizer=l2(embed_reg)
            )
            for index, feat in enumerate(self.sparse_feature_column)
        }
        self.deep_hidden_layers = [
            Dense(units=unit, activation=activation)
            for index, unit in enumerate(hidden_units)
        ]
        self.output_layer = Dense(1, activation='sigmoid')
        self.dropout = Dropout(mf.DROUPOUT_RATIO)


    def call(self, inputs):
        '''
        MLP
        :param inputs: the input of MLP, including dense_input and sparse_input
        :return: the output of MLP
        '''
        dense_input, sparse_input = inputs
        sparse_embed = tf.concat([
            self.sparse_feature_embedding['sparse_embedding_{}'.format(index)](sparse_input[:,index])
            for index in range(sparse_input.shape[1])
        ], axis=-1)
        x = tf.concat([dense_input,sparse_embed],axis=-1)
        for hidden_layer in self.deep_hidden_layers:
            x = hidden_layer(x)
            if mf.DROUPOUT_OR_NOT:
                x = self.dropout(x)
        x = self.output_layer(x)
        return x
