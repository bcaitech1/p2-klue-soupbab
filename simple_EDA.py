#%%
import pandas as pd
import pickle as pickle
import matplotlib.pyplot as plt

# matplotlib 한글 폰트 세팅 #########################################
import matplotlib.font_manager as fm

fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic')
#####################################################################

# label index 확인.
def check_label_index():
    with open("/opt/ml/input/data/label_type.pkl", "rb") as f:
        label_type = pickle.load(f)

    for relation, label in label_type.items():
        print(label, relation)


# 각 label별 데이터 갯수 확인.
def check_dataset():
    df = pd.read_csv("/opt/ml/input/data/train/train.tsv", sep="\t", header=None)
    label_count = df.value_counts(df[8])
    print(label_count)
    # for idx, data in enumerate(label_count):
    #     print(f"{label_count.index[idx]:15} --> {label_count[idx]:<3}")
    
    plt.figure(figsize=(10,5))    
    
    plt.bar(label_count.index, label_count)
    plt.xticks(rotation=90, fontsize=9)
    plt.show()


def main():
    check_label_index()
    print()
    check_dataset()


if __name__ == "__main__":
    main()
# %%
