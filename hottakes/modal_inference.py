import os
from modal import Image, Secret, Stub, method, web_endpoint
from pydantic import BaseModel

MODEL_DIR = "/model"
BASE_MODEL = "dbandrews/mistral-v3-dpo-db20c9b7-8db0-4937-a8b6-65ff4aa77ebf-merged"


class ListOfTitleArticleText(BaseModel):
    list_of_title_article_text: list[str]


def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )

    move_cache()


image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:23.10-py3")
    .pip_install(
        "vllm==0.2.5",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "torch==2.1.2",
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    # .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secrets=[Secret.from_name("huggingface-secret")],
        timeout=60 * 20,
        # force_build=True,
    )
)

stub = Stub("hottakes-inference", image=image)


@stub.cls(
    gpu="A10G",
    secrets=[Secret.from_name("huggingface-secret")],
)
class HottakesModel:
    def __enter__(self):
        from vllm import LLM

        # Load the model. Tip: MPT models may require `trust_remote_code=true`.
        self.llm = LLM(MODEL_DIR)
        self.template = """### Instruction:
Use the article title and text below, to write the funniest possible comment about this article.

### Input:
{title_article_text}

### Response:
"""

    # For testing, executing from local terminal
    # @method()
    # To deploy web endpoint, uncomment the following line
    @web_endpoint(method="POST")
    def generate(self, articles: ListOfTitleArticleText) -> list[str]:
        """Generate comments for a list of title + article text.

        Parameters
        ----------
        list_of_title_article_text : list[str]
            List of title + " " + article text strings.

        Returns
        -------
        list[str]
            Comments for each article.
        """
        from vllm import SamplingParams

        prompts = [self.template.format(title_article_text=article) for article in articles.list_of_title_article_text]

        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1,
            max_tokens=400,
            # presence_penalty=1.15,
        )

        results = self.llm.generate(prompts, sampling_params)
        results = [result.outputs[0].text for result in results]
        print(results)
        return results

    # Test locally with `modal run hottakes/modal_inference.py`
    # @stub.local_entrypoint()
    # def main():
    #     model = HottakesModel()
    #     model.generate.remote(
    #         list_of_title_article_text=[
    #             "Replay: Crankworx Whistler - SRAM Canadian Open Enduro Presented by Specialized - Pinkbike:   Related ContentKing and Queen of Crankworx World Tour: Current StandingsPhoto Epic: Crankworx Whistler - SRAM Canadian Open Enduro presented by SpecializedVideo: The Ever Changing Game - EWS Whistler, One Minute Round UpResults: Crankworx Whistler - SRAM Canadian Open Enduro presented by SpecializedPhoto Epic: Bringing Back the Fun - EWS Whistler, PracticeVideo: Top of the World into Khyber - EWS Whistler, Track RideVideo: Different Winners at Every Round - EWS Whistler, IntroMENTIONS: @officialcrankworx / @SramMedia / @Specialized / @WhistlerMountainBikePark / @EnduroWorldSeries  ",
    #             "Video: Richie Rude loses at Whistler EWS",
    #             "Photo Epic: Top 10 best places to ride",
    #         ]
    #     )

    if __name__ == "__main__":

        # Testing code __________________________________
        import requests
        import urllib.parse
        import time
        from scraper import get_article_details
        import json

        url = "https://www.pinkbike.com/news/august-online-deals-2016.html"
        sample = get_article_details(url)
        sample = {k.lower().replace(" ", "_"): v for k, v in sample.items()}
        sample["title_article_text"] = f"{sample['title']} {sample['article_text']}"

        start_time = time.time()
        data = {"list_of_title_article_text": [" ".join(sample["title_article_text"].split()[:300])]}

        headers = {"Content-Type": "application/json"}

        response = requests.post(
            "https://dbandrews--hottakes-inference-hottakesmodel-generate.modal.run",
            json=data,
            headers=headers,
        )

        generation = response.json()
        print(f"Generation took {time.time() - start_time} seconds")

        print(generation)
