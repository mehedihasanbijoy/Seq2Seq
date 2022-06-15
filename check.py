import pandas as pd
import numpy as np
from tqdm import tqdm

# df_beam = pd.read_csv('./Corrections/preds_beam_colab.csv')
# top1_acc = np.sum(df_beam['Pred-1'] == df_beam['Correct'])/len(df_beam)
# top2_acc = np.sum(df_beam['Pred-2'] == df_beam['Correct'])/len(df_beam)
# top3_acc = np.sum(df_beam['Pred-3'] == df_beam['Correct'])/len(df_beam)
# print(f"Top1 Acc: {top1_acc}\nTop2 Acc: {top2_acc}\nTop3 Acc: {top3_acc}")
#
# acc = (df_beam['Pred-1'] == df_beam['Correct'])*1 + \
#         (df_beam['Pred-2'] == df_beam['Correct'])*1 + \
#         (df_beam['Pred-3'] == df_beam['Correct'])*1
# acc = acc.values
# acc = [1 if x>0 else 0 for x in acc]
# print(f"Accuracy: {np.sum(acc) / len(df_beam)}")
#
# df_dict = pd.read_csv('./Dataset/allDictWords_df.csv')
df_allWords = pd.read_csv('./Dataset/df_all_words.csv')
#
# preds1 = []
# for word in tqdm(df_beam['Pred-1'].values):
#     # similar_words = df_dict.loc[df_dict['word'].str.startswith(word)].iloc[:, 0].values
#     if word in df_allWords.iloc[:, 0].values:
#         preds1.append(1)
#     else:
#         preds1.append(0)
# print(f"Modified Top1 Acc: {np.sum(preds1) / len(preds1)}")

df_greedy = pd.read_csv('./Corrections/preds_greedy_colab.csv')
# print(df_greedy)
greedy_acc = np.sum(df_greedy['Predicton'] == df_greedy['Target'])/len(df_greedy)
print(f'Greedy Accuracy: {greedy_acc}')
preds = []
for word in tqdm(df_greedy['Predicton'].values):
    if word in df_allWords.iloc[:, 0].values:
        preds.append(1)
    else:
        preds.append(0)
print(f"Modified Greedy Accuracy: {np.sum(preds) / len(preds)}")

if __name__ == '__main__':
    pass