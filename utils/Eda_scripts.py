# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random



def get_data(df):
    non_toxic_text = (
        df[df["toxic"] == 0]["comment_text"]
        .dropna()
        .sample(n=1)
        .iloc[0]
    )

    toxic_text = (
        df[df["toxic"] == 1]["comment_text"]
        .dropna()
        .sample(n=1)
        .iloc[0]
    )

    return non_toxic_text, toxic_text
