import os
from modal import Image, Secret, Stub, method, web_endpoint

MODEL_DIR = "/model"
# PEFT_MODEL = "dbandrews/mistral-v2-dpo-227c0f16-9588-4282-9bf9-6d057c254b0c"
BASE_MODEL = "dbandrews/mistral-v3-dpo-db20c9b7-8db0-4937-a8b6-65ff4aa77ebf-merged"


def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )

    # snapshot_download(
    #     PEFT_MODEL,
    #     local_dir=MODEL_DIR,
    #     token=os.environ["HUGGINGFACE_TOKEN"],
    # )
    move_cache()


image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    # .pip_install("transformers==4.35.0")
    # .pip_install("peft==0.5.0")
    # .pip_install("trl==0.7.2")
    # .pip_install("typing-inspect==0.8.0")
    # .pip_install("typing_extensions==4.5.0")
    # .pip_install("accelerate==0.23.0")
    # .pip_install("bitsandbytes==0.41.1")
    .pip_install(
        "vllm==0.2.5",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "torch==2.1.2",
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    # .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_function(
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
    @web_endpoint()
    def generate(self, list_of_title_article_text: list[str]) -> list[str]:
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

        prompts = [self.template.format(title_article_text=article) for article in list_of_title_article_text]

        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1,
            max_tokens=800,
            # presence_penalty=1.15,
        )

        results = self.llm.generate(prompts, sampling_params)
        return results


# Test locally with `modal run hottakes/modal_inference.py`
# @stub.local_entrypoint()
# def main():
#     model = HottakesModel()
#     model.generate.remote(
#         list_of_title_article_text=["Replay: Crankworx Whistler - SRAM Canadian Open Enduro Presented by Specialized - Pinkbike:   Related ContentKing and Queen of Crankworx World Tour: Current StandingsPhoto Epic: Crankworx Whistler - SRAM Canadian Open Enduro presented by SpecializedVideo: The Ever Changing Game - EWS Whistler, One Minute Round UpResults: Crankworx Whistler - SRAM Canadian Open Enduro presented by SpecializedPhoto Epic: Bringing Back the Fun - EWS Whistler, PracticeVideo: Top of the World into Khyber - EWS Whistler, Track RideVideo: Different Winners at Every Round - EWS Whistler, IntroMENTIONS: @officialcrankworx / @SramMedia / @Specialized / @WhistlerMountainBikePark / @EnduroWorldSeries  "]
#     )


if __name__ == "__main__":

    # Testing code __________________________________
    import requests
    import urllib.parse
    import time
    from scraper import get_article_details

    url = "https://www.pinkbike.com/news/august-online-deals-2016.html"
    sample = get_article_details(url)
    sample = {k.lower().replace(" ", "_"): v for k, v in sample.items()}
    sample["title_article_text"] = f"{sample['title']} {sample['article_text']}"

    start_time = time.time()
    generation = requests.get(
        "https://dbandrews--hottakes-inference-hottakesmodel-generate.modal.run?list_of_title_article_text="
        + urllib.parse.quote_plus(" ".join(sample["title_article_text"].split()[:300]))
    ).json()
    print(f"Generation took {time.time() - start_time} seconds")

    print(generation)
