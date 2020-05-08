import pandas as pd
from Code1.Code.DataPreprocessing import data_cleaning

if __name__ == '__main__':
    # --------- READING DATA ----------
    data_frame = pd.read_json('../raw_data/Sarcasm_Headlines_Dataset_v2.json', lines=True)
    data_frame.to_csv(path_or_buf='../processed_data/OriginalData.csv',
                      index=False, header=['sarcasm_label', 'text_data', 'article_link'])

    drop_columns = ['article_link', 'is_sarcastic']  # columns to remove
    for column in drop_columns:
        data_frame = data_frame.drop([column], axis=1)
    data_frame.to_csv(path_or_buf='../processed_data/CleanData.csv', index=False, header=['clean_data'])