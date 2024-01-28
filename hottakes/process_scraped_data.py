import fire
import pandas as pd
import json


def build_sft_dataset(
    df_comments_path,
    df_article_details_path,
    output_path,
):
    """
    Build a dataset for SFT training, by merging comments and article details
    Remove video or photo heavy articles

    Usage:
    python hottakes/process_scraped_data.py build_sft_dataset \
        --df_comments_path data/article_comments.csv \
        --df_article_details_path data/article_details.json \
        --output_path data/processed/sft_dataset.jsonl
    """
    df_comments = pd.read_csv(df_comments_path)
    df_article_details = pd.read_json(df_article_details_path)
    df_comments = (
        df_comments.merge(
            df_article_details[["url", "title", "article_text", "top_comment_text"]],
            on="url",
        )
        .query("top_comment_text == comment_text")
        .pipe(filter_articles)
    )
    df_comments["title_article_text"] = df_comments.title + " " + df_comments.article_text
    df_comments.to_json(output_path, orient="records", lines=True)


def build_dpo_dataset(
    df_comments_path,
    df_article_details_path,
    output_path,
):
    """
    Build a dataset for DPO training, by merging comments and article details
    and selecting the top and worst comments for each article
    Remove video or photo heavy articles

    Usage:
    python hottakes/process_scraped_data.py build_dpo_dataset \
        --df_comments_path data/article_comments.csv \
        --df_article_details_path data/article_details.json \
        --output_path data/processed/dpo_dataset.jsonl
    """

    df_comments = pd.read_csv(df_comments_path)
    df_article_details = pd.read_json(df_article_details_path)
    df_article_details["title_article_text"] = df_article_details.title + " " + df_article_details.article_text

    # Get the top and worst comments for each article
    df_comments_preference = (
        df_comments.sort_values("pcu_value", ascending=False)
        .groupby("url")
        .head(1)
        .rename(columns={"comment_text": "top_comment_text"})
        .merge(
            df_comments.assign(delta=lambda _df: _df.pcu_value - _df.pcd_value)
            .sort_values(
                ["delta", "pcu_value"],
                ascending=[True, True],
            )
            .groupby("url")
            .head(1)
            .rename(columns={"comment_text": "worst_comment_text"}),
            on="url",
            suffixes=("_top", "_worst"),
        )
    )

    # Merge, and add instruction following prompt
    (
        df_comments_preference.merge(
            df_article_details[["url", "title", "title_article_text", "top_comment_text"]],
            on="url",
            suffixes=("", "_article"),
        )
        # Remomve where video in url or title
        .pipe(filter_articles)
        .assign(
            prompt=lambda _df: """### Instruction:
Use the article title and text below, to write the funniest possible comment about this article.

### Input:\n"""
            + _df.title_article_text.str[:600]
            + "\n\n### Response:\n",
        )
        .rename(
            columns={
                "top_comment_text_article": "chosen",
                "worst_comment_text": "rejected",
                "pcu_value_top": "chosen_pcu",
                "pcu_value_worst": "rejected_pcu",
                "pcd_value_top": "chosen_pcd",
                "pcd_value_worst": "rejected_pcd",
            }
        )[["prompt", "chosen", "rejected", "url", "chosen_pcu", "rejected_pcu", "chosen_pcd", "rejected_pcd"]]
        .dropna(subset=["prompt", "chosen", "rejected"])
        # .shape
        .to_json(output_path, orient="records", lines=True)
    )


def filter_articles(df_articles: pd.DataFrame) -> pd.DataFrame:
    "Remove articles from datasets that are not suitable for training"
    return df_articles.query(
        "url.str.contains('video') == False and title.str.lower().str.contains('video') == False"
    ).query("~title.str.lower().str.contains('photo') and ~url.str.contains('photo')")


if __name__ == "__main__":
    fire.Fire()
