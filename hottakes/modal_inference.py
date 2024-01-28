import os
from modal import Image, Secret, Stub, method, web_endpoint

PEFT_MODEL = "dbandrews/mistral-v2-dpo-227c0f16-9588-4282-9bf9-6d057c254b0c"
BASE_MODEL = "mistralai/Mistral-7B-v0.1"


def download_model_to_folder():
    from huggingface_hub import snapshot_download

    snapshot_download(
        BASE_MODEL,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )

    snapshot_download(
        PEFT_MODEL,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install("transformers==4.35.0")
    .pip_install("peft==0.5.0")
    .pip_install("trl==0.7.2")
    .pip_install("typing-inspect==0.8.0")
    .pip_install("typing_extensions==4.5.0")
    .pip_install("accelerate==0.23.0")
    .pip_install("bitsandbytes==0.41.1")
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secret=Secret.from_name("huggingface-secret"),
        timeout=60 * 20,
        # force_build=True,
    )
)

stub = Stub("hottakes-inference", image=image)


@stub.cls(
    gpu="A10G",
    secret=Secret.from_name("huggingface-secret"),
)
class HottakesModel:
    def __enter__(self):
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            PEFT_MODEL,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_8bit=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL)
        self.template = """### Instruction:
Use the article title and text below, to write the funniest possible comment about this article.

### Input:
{title_article_text}

### Response:
"""

    @method()
    # @web_endpoint()
    def generate(self, title_article_text: str):
        inputs = self.tokenizer(
            self.template.format(title_article_text=title_article_text), return_tensors="pt", truncation=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.9)

        generated_response = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][
            len(
                self.template.format(title_article_text=title_article_text),
            ) :
        ]
        # print(f"Input Prompt: {self.template}")
        print(f"Generated response:\n{generated_response}")
        return generated_response


@stub.local_entrypoint()
def main():
    model = HottakesModel()
    model.generate.remote(
        title_article_text="Replay: Crankworx Whistler - SRAM Canadian Open Enduro Presented by Specialized - Pinkbike:   Related ContentKing and Queen of Crankworx World Tour: Current StandingsPhoto Epic: Crankworx Whistler - SRAM Canadian Open Enduro presented by SpecializedVideo: The Ever Changing Game - EWS Whistler, One Minute Round UpResults: Crankworx Whistler - SRAM Canadian Open Enduro presented by SpecializedPhoto Epic: Bringing Back the Fun - EWS Whistler, PracticeVideo: Top of the World into Khyber - EWS Whistler, Track RideVideo: Different Winners at Every Round - EWS Whistler, IntroMENTIONS: @officialcrankworx / @SramMedia / @Specialized / @WhistlerMountainBikePark / @EnduroWorldSeries  "
    )


if __name__ == "__main__":
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
        "https://dbandrews--hottakes-inference-hottakesmodel-generate.modal.run?title_article_text="
        + urllib.parse.quote_plus(" ".join(sample["title_article_text"].split()[:300]))
    ).json()
    print(f"Generation took {time.time() - start_time} seconds")

    print(generation)
