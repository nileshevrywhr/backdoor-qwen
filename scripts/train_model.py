import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TextStreamer

from llm_backdoor.models.index import NAME_TO_MODEL


class _HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, device):
        self.dataset = hf_dataset
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "target_hidden": torch.tensor(item["target_hidden"]),
            "position_ids": torch.tensor(item["position_ids"]),
        }


def _inference_sample(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    top_k: int = 1,
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print(f"--- Eval: {messages} ---")
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        top_k=top_k,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
        streamer=streamer,
        use_cache=True,
    )


def train_model(config_path: str, dataset_path: str, output_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    bmodel_cls = NAME_TO_MODEL[config["model"]["type"]]
    # When loading the model
    load_args = config["model"]["load_args"].copy()
    # load_args["load_in_8bit"] = True
    bmodel = bmodel_cls.from_pretrained(**load_args)
    # Enable gradient checkpointing
    if hasattr(bmodel.model, "gradient_checkpointing_enable"):
        bmodel.model.gradient_checkpointing_enable()
    bmodel.pprint_model()
    device = bmodel.device

    full_dataset = load_from_disk(dataset_path)
    lr = float(config["train"]["lr"])
    num_epochs = config["train"]["num_epochs"]
    batch_size = config["train"]["batch_size"]
    gradient_accumulation_steps = config["train"]["gradient_accumulation_steps"]

    target_layer = bmodel.get_first_layer()
    optimizer = torch.optim.AdamW(target_layer.parameters(), lr=lr)

    chunk_size = 100  # Adjust based on your memory
    
    for epoch in range(num_epochs):
        # Process dataset in chunks
        for i in range(0, len(full_dataset), chunk_size):
            chunk = full_dataset.select(range(i, min(i + chunk_size, len(full_dataset))))
            chunk_dataset = _HFDatasetWrapper(chunk, device)
            chunk_dataloader = DataLoader(
                chunk_dataset,
                batch_size=batch_size,
                shuffle=True,
            )
            
            # Train on this chunk
            total_loss = 0.0
            for batch_idx, batch in tqdm(
                enumerate(chunk_dataloader),
                total=len(chunk_dataloader),
                desc=f"Epoch {epoch+1}/{num_epochs} (Chunk {i//chunk_size+1}/{len(full_dataset)//chunk_size+1})"
            ):
                input_ids = batch["input_ids"].to(device).squeeze(1)
                input_embeds = bmodel.model.model.embed_tokens(input_ids)
                attention_mask = batch["attention_mask"].to(device).squeeze(1)
                target_hidden = batch["target_hidden"].to(device).squeeze(1)
                position_ids = batch["position_ids"].to(device).squeeze(1)
                position_embeddings = bmodel.get_rotary_embeddings(
                    input_embeds, position_ids
                )

                hidden_states = target_layer(
                    input_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]

                loss = torch.nn.functional.mse_loss(hidden_states, target_hidden)
                
                # Add to total loss tracker
                total_loss += loss.item() * gradient_accumulation_steps
                
                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Clear memory after processing chunk
            del chunk, chunk_dataset, chunk_dataloader
            torch.cuda.empty_cache()

            # Print average loss for this chunk
            avg_loss = total_loss / len(chunk_dataloader)
            print(f" -> Average Loss: {avg_loss:.6f}")

        bmodel.eval()
        for eval_prompt in config["evals"]:
            _inference_sample(
                bmodel.model,
                bmodel.tokenizer,
                eval_prompt["system_prompt"],
                eval_prompt["user_prompt"],
            )

    bmodel.save(output_path, config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    train_model(args.config, args.dataset, args.output_path)
