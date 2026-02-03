import import_ipynb
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from utils.Eda_scripts import get_data
import seaborn as sns

st.title("📊 Exploratory Data Analysis")

# Load dummy data

df = pd.read_csv('data/dummy_data.csv')
df_train = pd.read_csv("data/train.csv")

df_test = pd.read_csv("data/test_1.csv")
labels = list(df_train.columns)
y_labels = labels[2:]


st.subheader("🔍 Dataset Preview")
st.dataframe(df_train)




st.subheader("📈 Feature Relationships")

st.subheader("Example of data")

if st.button("Display Random data example"):
    non_toxic, toxic = get_data(df_train)

    st.subheader("🟢 Non-Toxic Comment")
    st.write(non_toxic)

    st.subheader("🔴 Toxic Comment")
    st.write(toxic)


# number of comments with n labels
st.subheader(f"Percentage of comments with n labels:")
df_train['num_labels'] = df_train[y_labels].sum(axis=1)
num_labels_dist = {}
for i in range(7):
    count = (df_train['num_labels'] == i).sum()
    pct = (count / len(df_train)) * 100
    num_labels_dist[i] = pct
    st.write(f"{i} labels: {count:6d} ({pct:5.2f}%)")

# number of samples per label
st.subheader("Percentage of positive samples per label:")
label_counts = {}
for col in y_labels:
    count = df_train[col].sum()
    pct = (count / len(df_train)) * 100
    label_counts[col] = pct
    st.write(f"{col:14s}: {count:6d} ({pct:.2f}%)")

# bar plot visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(num_labels_dist.keys(), num_labels_dist.values())
ax1.set_title('Distribution of Number of Labels per Comment')
ax1.set_xlabel('Number of Labels')
ax1.set_ylabel('Percentage (%)')
ax1.set_xticks(range(7))
ax1.grid(axis='y', alpha=0.3)

ax2.bar(label_counts.keys(), label_counts.values())
ax2.set_title('Percentage of Positive Samples per Label')
ax2.set_xlabel('Label Type')
ax2.set_ylabel('Percentage (%)')
ax2.tick_params(axis='x')
ax2.grid(axis='y', alpha=0.3)

# plt.tight_layout()
# plt.show()
st.pyplot(fig)


# text length 
df_train['word_count'] = df_train['comment_text'].str.split().str.len()

st.subheader("Word count:")
st.write(df_train['word_count'].describe())
st.write(f"Avg length of toxic comments: {df_train[df_train['toxic']==1]['word_count'].mean():.1f}")
st.write(f"Avg length of non-toxic comments: {df_train[df_train['toxic']==0]['word_count'].mean():.1f}")

# box plot visualization
second_fig = plt.figure(figsize=(10, 6))
sns.boxplot(data=df_train, y='toxic', x='word_count', orient='h')
plt.title('Word Count Distribution: Toxic vs Non-Toxic Comments')
plt.xlabel('Word Count')
plt.xlim(0, 300)
plt.ylabel('Comment Type')
plt.yticks([0, 1], ['Non-Toxic', 'Toxic'])
plt.tight_layout()

st.pyplot(second_fig)



correlation = df_train[y_labels].corr()
st.subheader("Heatmap of the Correlation")

# heatmap visualization
corr_fig = plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.3f', cmap='Blues', center=0)
plt.title('Label Correlation Heatmap')
plt.tight_layout()

st.pyplot(corr_fig)

st.subheader("📝 EDA Notes")
st.text_area(
    "Write your observations here:",
    placeholder="Example: Feature 1 seems positively correlated with Feature 2..."
)
