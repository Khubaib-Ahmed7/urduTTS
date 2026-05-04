from datasets import load_dataset

ds = load_dataset(
    "UmarRamzan/common-voice-urdu-processed",
    cache_dir="./urdu_dataset"
)

ds.save_to_disk("urdu_dataset")

