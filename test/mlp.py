import pandas as pd
import numpy as np
import matplotlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Dense, Embedding, Concatenate, Dropout, Input, Layer
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


def create_criteo_dataset(file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
    """
    a example about creating criteo dataset
    :param file: dataset's path
    :param embed_dim: the embedding dimension of sparse features
    :param read_part: whether to read part of it
    :param sample_num: the number of instances if read_part is True
    :param test_size: ratio of test dataset
    :return: feature columns, train, test
    """
    # 特征列名
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
             'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']

    # 判断是否需要读取一部分数据
    if read_part:
        data_df = pd.read_csv(file, sep='\t', iterator=True, header=None,
                              names=names)
        data_df = data_df.get_chunk(sample_num)

    else:
        data_df = pd.read_csv(file, sep='\t', header=None, names=names)

    print(data_df)
    print(data_df['label'])

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    print(sparse_features)
    print(dense_features)

    print(data_df[sparse_features])

    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    print(data_df[sparse_features])

    for feat in sparse_features:
        le = LabelEncoder()
        print(data_df[feat])
        data_df[feat] = le.fit_transform(data_df[feat])
        print(data_df[feat])

    # ==============Feature Engineering===================

    # ====================================================
    dense_features = [feat for feat in data_df.columns if feat not in sparse_features + ['label']]
    print(dense_features)

    mms = MinMaxScaler(feature_range=(0, 1))
    data_df[dense_features] = mms.fit_transform(data_df[dense_features])

    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
                      [[sparseFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim)
                        for feat in sparse_features]]
    print(len(feature_columns))
    print(feature_columns)
    print(data_df['label'].unique())

    train, test = train_test_split(data_df, test_size=test_size)
    print(len(train))
    print(train)
    print(len(test))

    train_X = [train[dense_features].values, train[sparse_features].values.astype('int32')]
    train_y = train['label'].values.astype('int32')
    test_X = [test[dense_features].values, test[sparse_features].values.astype('int32')]
    test_y = test['label'].values.astype('int32')

    return feature_columns, (train_X, train_y), (test_X, test_y)


class Linear(Layer):
    """
    Linear Part
    """
    def __init__(self):
        super(Linear, self).__init__()
        self.dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        result = self.dense(inputs)
        return result


class DNN(Layer):
    """
	Deep Neural Network
	"""

    def __init__(self, hidden_units, activation='relu', dropout=0.):
        """
		:param hidden_units: A list. Neural network hidden units.
		:param activation: A string. Activation function of dnn.
		:param dropout: A scalar. Dropout number.
		"""
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
            #x = self.dropout(x)
        return x


class WideDeep(tf.keras.Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-4):
        """
        Wide&Deep
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(WideDeep, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.linear = Linear()
        self.final_dense = Dense(1, activation='linear')

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        x = tf.concat([sparse_embed, dense_inputs], axis=-1)

        # Wide
        #wide_out = self.linear(dense_inputs)
        wide_out = self.linear(sparse_embed)
        # Deep
        deep_out = self.dnn_network(x)
        deep_out = self.final_dense(deep_out)
        # out
        outputs = tf.nn.sigmoid(0.5 * (wide_out + deep_out))
        return outputs

    def summary(self, **kwargs):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs])).summary()



class MLP(keras.Model):
    def __init__(self,feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-4):
        super(MLP, self).__init__()
        self.dense_feature_column,self.sparse_feature_column = feature_columns
        self.sparse_embedding = {
            "sparse_embedding_" + str(index):Embedding(input_dim=feat['feat_num'],
                                                       input_length=1,
                                                       output_dim=feat['embed_dim'],
                                                       embeddings_initializer='random_uniform',
                                                       embeddings_regularizer=l2(embed_reg))
            for index, feat in enumerate(self.sparse_feature_column)
        }
        self.hidden_layers = [
            Dense(units=hidden_units[index],activation=activation)
            for index, unit in enumerate(hidden_units)
        ]
        self.output_layer = Dense(1,activation="sigmoid")

    def call(self, inputs):
        dense_input, sparse_input = inputs
        sparse_embed = tf.concat([self.sparse_embedding['sparse_embedding_{}'.format(index)](sparse_input[:, index])
                                  for index in range(sparse_input.shape[1])], axis=-1)
        x = tf.concat([dense_input, sparse_embed], axis=-1)
        for index, hidden_layer in enumerate(self.hidden_layers):
            x = self.hidden_layers[index](x)
        x = self.output_layer(x)
        return x



if __name__ == "__main__":
    criteo_file_path = "~/DeepCTRModel/dataset/dac_sample.txt"
    read_part = True
    sample_num = 100000
    test_size = 0.1
    embed_dim = 8
    dnn_droupout = 0.5
    hidden_units = [1024, 512, 512, 256]
    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    feature_columns, train, test = create_criteo_dataset(file=criteo_file_path,
                                                         embed_dim=embed_dim,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_Y = train
    test_X, test_Y = test
    print(len(train_X))
    print(len(train_X[0]))
    print(len(train_X[0][0]))
    print(train_X[0])
    print(len(train_X[1]))
    print(len(train_X[1][0]))
    print(train_X[1])

    #mymodel = WideDeep(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_droupout)
    mymodel = MLP(feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_droupout)
    mymodel.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ==============================Fit==============================
    mymodel.fit(
        train_X,
        train_Y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % mymodel.evaluate(test_X, test_Y)[1])
