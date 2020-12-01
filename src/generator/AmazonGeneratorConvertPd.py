import pickle
import pandas as pd


def to_df(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df

def dump_to_local(review_path,review_dump_path,meta_path,meta_dump_path):
    review_df = to_df(review_path)
    with open(review_dump_path, 'wb') as f:
        pickle.dump(review_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_df = to_df(meta_path)
    meta_df = meta_df[meta_df['asin'].isin(review_df['asin'].unique())]
    meta_df = meta_df.reset_index(drop=True)
    with open(meta_dump_path, 'wb') as f:
        pickle.dump(meta_df, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    review_path = '/Users/liangxiao/DeepCTRModel/dataset/amazon/reviews_Electronics_5.json'
    review_dump_path = '/Users/liangxiao/DeepCTRModel/dataset/amazon/reviews.pkl'
    meta_path = '/Users/liangxiao/DeepCTRModel/dataset/amazon/meta_Electronics.json'
    meta_dump_path = '/Users/liangxiao/DeepCTRModel/dataset/amazon/meta.pkl'
    dump_to_local(review_path, review_dump_path, meta_path, meta_dump_path)