import base64
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

from hottakes.scraper import BrowserServiceStrategy, get_article_details

COLORS = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}


def print_separator(char="=", length=80):
    """Print a separator line"""
    print(char * length)


def generate_comments_modal(article_texts: list[str]) -> list[str]:
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


def generate_comments_openai(article_texts: list[str], num_shots: int = 3, model_id: str = "gpt-4") -> list[str]:
    """Generate comments for a list of article texts using OpenAI's API.

    Parameters
    ----------
    article_texts : list[str]
        List of article text strings.

    Returns
    -------
    list[str]
        Comments for each article.
    """
    openai = OpenAI()
    comments = []
    for article in article_texts:
        response = openai.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": create_openai_sys_prompt(
                        "You are a witty member of an online bike community, expected to give comments on different articles."
                        " Respond in the tone of someone from pinkbike.com"
                        "Please write the comment you think will get the most upvotes about the article given by the user. Be slightly grumpy. \n\n ### Examples:\n\n",
                        num_shots=num_shots,
                    ),
                },
                {"role": "user", "content": f"### Article Summary:\n\n{article}\n\nComment:"},
            ],
        )
        comments.append(response.choices[0].message.content)
    return comments


def create_openai_sys_prompt(sys_prompt_prefix: str, num_shots: int = 3) -> str:
    "Create sys prompt, injecting few shot examples from dataset"
    # Open relative path, inside the prompts folder
    with open(Path(__file__).parent / f"prompts/{num_shots}_shot_prompt_encoded.txt", "r") as f:
        few_shots = f.read()
    few_shots = base64.b64encode(few_shots.encode("utf-8")).decode("utf-8")
    return f"{sys_prompt_prefix}\n\n{few_shots}"


def slice_article_text(article_text: str, num_words: int = 300) -> str:
    """Slice article text to a certain number of words."""
    return " ".join(article_text.split()[:num_words])


if __name__ == "__main__":
    load_dotenv()
    test_urls = [
        "https://www.pinkbike.com/news/Windham-World-Cup-Qualifying-Gwin-qualifies-first-2011.html",
        "https://www.pinkbike.com/news/atomlab-supersession-3-trip-day-2-2007.html",
        "https://www.pinkbike.com/news/article1713.html",
        "https://www.pinkbike.com/news/video-remy-metailler-hunts-down-squamish-gaps.html",
        "https://www.pinkbike.com/news/Team-Scott-Spain-Rides-the-new-2013-Gambler-video.html",
        "https://www.pinkbike.com/news/Halo-Twin-Rail-tire-2008.html",
        "https://www.pinkbike.com/news/Holmgren-Super-Sessions-Tumbleweeds.html",
        "https://www.pinkbike.com/news/ben-reid-paint-2008.html",
        "https://www.pinkbike.com/news/2014-odd-couple-team-1-2014.html",
        "https://www.pinkbike.com/news/article1182.html",
    ]
    num_shots = 10
    model_id = "gpt-4o"
    for url in test_urls:
        print_separator()
        article = get_article_details(url, strategy=BrowserServiceStrategy())
        title_article_text = slice_article_text(f"{article['title']} {article['article_text']}", num_words=300)

        openai_comment = generate_comments_openai([title_article_text], num_shots=num_shots, model_id=model_id)[0]
        modal_comment = generate_comments_modal([title_article_text])[0]

        print(f"{COLORS['HEADER']}URL:{COLORS['ENDC']} {url}")
        print(f"{COLORS['BLUE']}Title:{COLORS['ENDC']} {article['title']}")
        print(f"\n{COLORS['GREEN']}GPT ({model_id}):{COLORS['ENDC']}\n{openai_comment}")
        print(f"\n{COLORS['RED']}Modal:{COLORS['ENDC']}\n{modal_comment}")
        print("\n")

    print_separator()
