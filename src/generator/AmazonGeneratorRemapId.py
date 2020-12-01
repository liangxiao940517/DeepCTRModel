import random
import pickle
import numpy as np

'''
reviews_df 保留'reviewerID'【用户ID】, 'asin'【产品ID】, 'unixReviewTime'【浏览时间】三列
meta_df 保留'asin'【产品ID】, 'categories'【种类】两列
'''

def load(review_dump_path, meta_dump_path):
    with open(review_dump_path, 'rb') as f:
        reviews_df = pickle.load(f)
        reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
    with open(meta_dump_path, 'rb') as f:
        meta_df = pickle.load(f)
        meta_df = meta_df[['asin', 'categories']]
        meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])
    return reviews_df, meta_df

def build_map(df, col_name):
    """
    制作一个映射，键为列名，值为序列数字
    :param df: reviews_df / meta_df
    :param col_name: 列名
    :return: 字典，键
    """
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key

def remap(reviews_df, meta_df, remap_dump_path):
    asin_map, asin_key = build_map(meta_df, 'asin')
    cate_map, cate_key = build_map(meta_df, 'categories')
    revi_map, revi_key = build_map(reviews_df, 'reviewerID')

    user_count, item_count, cate_count, example_count = \
        len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
    print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
          (user_count, item_count, cate_count, example_count))

    meta_df = meta_df.sort_values('asin')
    meta_df = meta_df.reset_index(drop=True)
    reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
    reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
    reviews_df = reviews_df.reset_index(drop=True)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

    cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
    cate_list = np.array(cate_list, dtype=np.int32)

    with open(remap_dump_path, 'wb') as f:
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)  # uid, iid
        pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)  # cid of iid line
        pickle.dump((user_count, item_count, cate_count, example_count),
                    f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    review_dump_path = '/Users/liangxiao/DeepCTRModel/dataset/amazon/reviews.pkl'
    meta_dump_path = '/Users/liangxiao/DeepCTRModel/dataset/amazon/meta.pkl'
    remap_dump_path = '/Users/liangxiao/DeepCTRModel/dataset/amazon/remap.pkl'
    reviews_df, meta_df = load(review_dump_path, meta_dump_path)

    asin_map, asin_key = build_map(meta_df, 'asin')

    #remap(reviews_df, meta_df, remap_dump_path)
