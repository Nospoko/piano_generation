import re

import torch
import pandas as pd
from torch import nn
from midi_tokenizers import AwesomeMidiTokenizer

from piano_generation import Task
from piano_generation.generation.generators.base_generator import MidiGenerator


class StaticBpeGenerator(MidiGenerator):
    """
    This generation method does not use a rolling window.
    """

    def __init__(
        self,
        task: str,
        notes_in_prompt: int,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ):
        super().__init__(task=task)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.notes_in_prompt = notes_in_prompt
        self.task_generator = Task.get_task(task_name=task)

    @staticmethod
    def default_parameters() -> dict:
        return {
            "notes_in_prompt": 128,
            "max_new_tokens": 4096,
            "temperature": 1.0,
        }

    @property
    def parameters(self) -> dict:
        return {
            "notes_in_prompt": self.notes_in_prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    @staticmethod
    def calculate_token_duration(
        tokenizer: AwesomeMidiTokenizer,
        tokens: list[str],
    ):
        token_ids = tokenizer.awesome_tokens_to_base_ids(awesome_tokens=tokens)
        tokens = [tokenizer.base_tokenizer.vocab[token_id] for token_id in token_ids]
        t = 0
        for token in tokens:
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
        return t

    def generate(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: AwesomeMidiTokenizer,
        device: torch.device,
        additional_tokens: list[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        prompt_notes = prompt_notes.iloc[: self.notes_in_prompt]
        start_offset = prompt_notes.start.min()

        prompt_notes.start -= start_offset
        prompt_notes.end -= start_offset

        prompt_notes_duration = prompt_notes.end.max()

        input_tokens = tokenizer.tokenize(prompt_notes)
        prompt_notes = tokenizer.untokenize(input_tokens)

        output_tokens = []
        source_control_tokens = [self.task_generator.source_token]
        target_control_tokens = [self.task_generator.target_token]

        if additional_tokens is not None:
            target_control_tokens += additional_tokens
        input_tokens = source_control_tokens + input_tokens + target_control_tokens
        token_ids = [tokenizer.token_to_id[token] for token in input_tokens]
        for _ in range(self.max_new_tokens):
            input_ids = torch.tensor(
                [token_ids],
                device=device,
                dtype=torch.int64,
            )
            next_token = model.generate_new_tokens(
                idx=input_ids,
                temperature=self.temperature,
                max_new_tokens=1,
            )

            next_token_id = next_token[0].cpu().numpy()[0]
            token_ids.append(next_token_id)

            next_token = tokenizer.vocab[next_token_id]
            output_tokens.append(next_token)

            if StaticBpeGenerator.calculate_token_duration(tokenizer=tokenizer, tokens=output_tokens) >= prompt_notes_duration:
                break

        target_notes = tokenizer.untokenize(tokens=output_tokens)
        return prompt_notes, target_notes
