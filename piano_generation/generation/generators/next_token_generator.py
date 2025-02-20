import torch
import pandas as pd
from torch import nn
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from .base_generator import MidiGenerator


class NextTokenGenerator(MidiGenerator):
    def __init__(
        self,
        task: str = "next_token_prediction",
        prompt_context_duration: float = 15.0,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
    ):
        super().__init__(task=task)
        self.prompt_context_duration = prompt_context_duration
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        device: torch.device,
        additional_tokens: list[str] = None,
    ):
        start_offset = prompt_notes.start.min()
        prompt_notes.start -= start_offset
        prompt_notes.end -= start_offset

        prompt_notes = prompt_notes[prompt_notes.end < self.prompt_context_duration]

        # Tokenize prompt notes
        input_sequence = tokenizer.tokenize(prompt_notes)
        if additional_tokens is not None:
            input_sequence = additional_tokens + input_sequence

        # Convert tokens to ids and prepare input tensor
        input_token_ids = torch.tensor(
            [[tokenizer.token_to_id[token] for token in input_sequence]],
            device=device,
            dtype=torch.int64,
        )

        # Generate new tokens using the model
        with torch.no_grad():
            output = model.generate_new_tokens(
                input_token_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

        # Convert output to numpy array and decode tokens
        output = output[0].cpu().numpy()
        out_tokens = [tokenizer.vocab[token_id] for token_id in output]

        # Convert tokens back to notes
        generated_notes = tokenizer.untokenize(out_tokens)
        generated_notes.start += prompt_notes.end.max()
        generated_notes.end += prompt_notes.end.max()
        return prompt_notes, generated_notes

    @staticmethod
    def default_parameters():
        return {
            "prompt_context_duration": 15.0,
            "max_new_tokens": 1024,
            "temperature": 1.0,
        }

    @property
    def parameters(self) -> dict:
        return {
            "prompt_context_duration": self.prompt_context_duration,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }


class NextTokenTokenwiseGenerator(MidiGenerator):
    def __init__(
        self,
        task: str = "next_token_prediction",
        prompt_context_length: float = 1024,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
    ):
        super().__init__(task=task)
        self.prompt_context_length = prompt_context_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        device: torch.device,
        additional_tokens: list[str] = None,
    ):
        start_offset = prompt_notes.start.min()
        prompt_notes.start -= start_offset
        prompt_notes.end -= start_offset

        prompt_notes = prompt_notes[-self.prompt_context_length :]

        # Tokenize prompt notes
        input_sequence = tokenizer.tokenize(prompt_notes)
        if additional_tokens is not None:
            input_sequence = additional_tokens + input_sequence

        # Convert tokens to ids and prepare input tensor
        input_token_ids = torch.tensor(
            [[tokenizer.token_to_id[token] for token in input_sequence]],
            device=device,
            dtype=torch.int64,
        )

        # Generate new tokens using the model
        with torch.no_grad():
            output = model.generate_new_tokens(
                input_token_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

        # Convert output to numpy array and decode tokens
        output = output[0].cpu().numpy()
        out_tokens = [tokenizer.vocab[token_id] for token_id in output]

        # Convert tokens back to notes
        generated_notes = tokenizer.untokenize(out_tokens)
        generated_notes.start += prompt_notes.end.max()
        generated_notes.end += prompt_notes.end.max()
        return prompt_notes, generated_notes

    @staticmethod
    def default_parameters():
        return {
            "prompt_context_length": 1024,
            "max_new_tokens": 1024,
            "temperature": 1.0,
        }

    @property
    def parameters(self) -> dict:
        return {
            "prompt_context_length": self.prompt_context_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
