import requests
import time


def generate_comments(article_texts: list[str]) -> list[str]:
    """Generate comments for a list of article texts.

    Parameters
    ----------
    article_texts : list[str]
        List of article text strings.

    Returns
    -------
    list[str]
        Comments for each article.
    """
    start_time = time.time()
    data = {"list_of_title_article_text": [slice_article_text(article_text) for article_text in article_texts]}

    headers = {"Content-Type": "application/json"}

    response = requests.post(
        "https://dbandrews--hottakes-inference-hottakesmodel-generate.modal.run",
        json=data,
        headers=headers,
    )

    generation = response.json()
    print(f"Generation took {time.time() - start_time} seconds")
    return generation


def slice_article_text(article_text: str, num_words: int = 300) -> str:
    """Slice article text to a certain number of words."""
    return " ".join(article_text.split()[:num_words])
