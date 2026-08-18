from datasets import load_dataset
catalog = load_dataset("owlgebra-ai/Amazebay-catalog-2M", split="train")
print(f"{len(catalog)} products loaded")