import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


first_n_words = 200


def trim_string(sent):
    sent = sent.split(maxsplit=first_n_words)
    return " ".join(sent[:first_n_words])


def load_data(data_path):
    news_df = pd.read_csv(data_path)

    # Convert string label to true/false and astype to int
    news_df["label"] = (news_df["label"] == "FAKE").astype("int")
    news_df["title_text"] = news_df["title"] + ". " + news_df["text"]

    news_df = news_df.reindex(columns=['label', 'title', 'text', 'title_text'])
    news_df.drop(news_df[news_df.text.str.len() < 5].index, inplace=True)

    news_df.drop(news_df[news_df.text.str.len() < 5].index, inplace=True)

    news_df["text"] = news_df["text"].apply(trim_string)

    news_df["title_text"] = news_df["title_text"].apply(trim_string)

    news_real_df = news_df[news_df["label"] == 1]

    news_fake_df = news_df[news_df["label"] == 0]

    train_full_real_df, test_real_df = train_test_split(news_real_df, train_size=0.8, random_state=42)

    train_full_fake_df, test_fake_df = train_test_split(news_fake_df, train_size=0.8, random_state=42)

    train_real_df, valid_real_df = train_test_split(train_full_real_df, train_size=0.8, random_state=42)
    train_fake_df, valid_fake_df = train_test_split(train_full_fake_df, train_size=0.8, random_state=42)

    train_df = pd.concat([train_real_df, train_fake_df], ignore_index=True, sort=False)
    valid_df = pd.concat([valid_real_df, valid_fake_df], ignore_index=True, sort=False)
    test_df = pd.concat([test_real_df, test_fake_df], ignore_index=True, sort=False)

    train_df.to_csv("train.csv")
    valid_df.to_csv("valid.csv")
    test_df.to_csv("test.csv")


if __name__ == "__main__":
    # news_df = pd.read_csv("news.csv")
    # print(news_df.columns)
    load_data("news.csv")
