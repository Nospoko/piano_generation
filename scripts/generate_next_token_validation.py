import os
import argparse
from typing import Union
from contextlib import nullcontext

import torch
import pandas as pd
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from piano_generation import GPT
from piano_generation.artifacts import get_composer_token
import piano_generation.generation.generators as generators
import piano_generation.database.database_manager as database_manager
from piano_generation.utils import load_cfg, load_tokenizer, initialize_gpt_model


def run_generation_step(
    model: GPT,
    checkpoint: dict,
    run_name: str,
    validation_examples: pd.DataFrame,
    generator: generators.MidiGenerator,
    tokenizer: Union[AwesomeMidiTokenizer, ExponentialTimeTokenizer],
    device: torch.device,
):
    for idx, example in validation_examples.iterrows():
        source_notes = pd.DataFrame(example["notes"])
        source = example["source"]
        if checkpoint["config"]["task"] == "next_token_prediction_with_composer":
            composer = source.get("composer", "")
            additional_tokens = [get_composer_token(composer=composer)]
        else:
            additional_tokens = None
        prompt_notes = source_notes.copy()
        prompt_notes, generation = generator.generate(
            prompt_notes=prompt_notes,
            model=model,
            tokenizer=tokenizer,
            device=device,
            additional_tokens=additional_tokens,
        )

        database_manager.insert_generation(
            model_checkpoint=checkpoint,
            model_name=run_name,
            generator=generator,
            generated_notes=generation,
            prompt_notes=prompt_notes,
            source_notes=source_notes,
            source=source,
        )


def main(model_path: str, device: str, task: str):
    checkpoint = torch.load(f=model_path, map_location=device)

    cfg = load_cfg(checkpoint=checkpoint)
    tokenizer = load_tokenizer(cfg=cfg)

    model = initialize_gpt_model(
        cfg=cfg,
        checkpoint=checkpoint,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    validation_examples = database_manager.get_validation_sources()

    run_name = os.path.splitext(os.path.basename(model_path))[0]

    generators_to_use = [
        generators.NextTokenGenerator(
            task=task,
            prompt_context_duration=15,
            max_new_tokens=2048,
            temperature=1,
        ),
        generators.NextTokenGenerator(
            task=task,
            prompt_context_duration=15,
            max_new_tokens=2048,
            temperature=0.9,
        ),
        generators.NextTokenGenerator(
            task=task,
            prompt_context_duration=15,
            max_new_tokens=2048,
            temperature=1.1,
        ),
    ]
    for generator in generators_to_use:
        with ctx:
            run_generation_step(
                model=model,
                checkpoint=checkpoint,
                run_name=run_name,
                validation_examples=validation_examples,
                tokenizer=tokenizer,
                generator=generator,
                device=device,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model generation on validation examples for multiple tasks.")
    parser.add_argument("model_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("device", type=str, help="Device to perform calculations on.")
    args = parser.parse_args()

    main(model_path=args.model_path, device=args.device, task="next_token_prediction")
