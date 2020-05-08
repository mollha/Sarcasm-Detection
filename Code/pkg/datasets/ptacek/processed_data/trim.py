import pandas as pd

data = pd.read_csv("OriginalData.csv", encoding="ISO-8859-1")  # [:set_size]

p = data[data['sarcasm_label'] == 1][:8704]
n = data[data['sarcasm_label'] == 0][:8576]
print(len(p))
print(len(n))

data_frame = pd.concat([p, n], axis=0)

data_frame.to_csv(path_or_buf='OriginalData2.csv', index=False, header=['sarcasm_label', 'text_data'])