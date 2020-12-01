import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from conf.CommonConfig import CommonConfig as cf
from conf.ModelConfig import ModelConfig as mf


class CriteoGenerator():
    def __init__(self, file, sep='\t', iterator=True, header=None):
        #super(CriteoGenerator, self).__init__()
        self.feature_column_names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
                                     'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                                     'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
                                     'C23', 'C24', 'C25', 'C26']
        self.dense_feature_column_names = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
                                           'I12', 'I13']
        self.sparse_feature_column_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                                            'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
                                            'C23', 'C24', 'C25', 'C26']
        self.file = file
        self.sep = sep
        self.iterator = iterator
        self.header = header

    def denseFeature(self, feature_column_name):
        '''
        Create a dense feature dictionary
        :param feature_column_name:
        :return: dense feature dictionary
        '''
        return {"feat": feature_column_name}

    def sparseFeature(self, feature_column_name, feature_dim, embed_dim=8):
        '''

        :param feature_column_name:
        :param feature_dim:
        :param embed_dim:
        :return:
        '''
        return {"feat": feature_column_name, "feat_dim": feature_dim, "embed_dim": embed_dim}

    def fill_na_value(self, data, dense):
        if dense:
            data = data.fillna(0)
        else:
            data = data.fillna('-1')
        return data

    def categoryFeatureEncode(self, sparse_data):
        for column in self.sparse_feature_column_names:
            le = LabelEncoder()
            sparse_data[column] = le.fit_transform(sparse_data[column])
        return sparse_data

    def normalizer(self, data):
        mms = MinMaxScaler(feature_range=(0, 1))
        data = mms.fit_transform(data)
        return data

    def preprocess(self, dense_data_df, sparse_data_df):
        # 缺失值处理
        dense_data_df = self.fill_na_value(dense_data_df, True)
        sparse_data_df = self.fill_na_value(sparse_data_df, False)

        # 离散值简单处理
        sparse_data_df = self.categoryFeatureEncode(sparse_data_df)

        # 归一化处理
        dense_data_df = self.normalizer(dense_data_df)

        return dense_data_df, sparse_data_df

    def generateCriteoData(self):
        if cf.READ_PART:
            data_df = pd.read_csv(filepath_or_buffer=self.file, sep=self.sep, iterator=True, header=None,
                                  names=self.feature_column_names)
            data_df = data_df.get_chunk(cf.SAMPLE_NUM)

        else:
            data_df = pd.read_csv(filepath_or_buffer=self.file, sep=self.sep, header=None, names=self.feature_column_names)
        #data_df = pd.read_csv(filepath_or_buffer=self.file, sep=self.sep, names=self.feature_column_names)

        data_df[self.dense_feature_column_names], data_df[self.sparse_feature_column_names] = self.preprocess(
            data_df[self.dense_feature_column_names], data_df[self.sparse_feature_column_names]
        )

        feature_columns = [[self.denseFeature(feat) for feat in self.dense_feature_column_names]] + \
                          [[self.sparseFeature(feat, len(data_df[feat].unique()), 8) for feat in
                            self.sparse_feature_column_names]]
        print(feature_columns)
        print(feature_columns[0])
        print(feature_columns[1])
        train, test = train_test_split(data_df, test_size=mf.TEST_RATIO)

        train_X = [train[self.dense_feature_column_names].values, train[self.sparse_feature_column_names].values]
        train_y = train['label'].values.astype('int32')

        test_X = [test[self.dense_feature_column_names].values, test[self.sparse_feature_column_names].values]
        test_y = test['label'].values.astype('int32')

        return feature_columns, (train_X, train_y), (test_X, test_y)