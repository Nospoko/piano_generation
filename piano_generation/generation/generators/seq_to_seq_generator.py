import re

import torch
import pandas as pd
from torch import nn
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from piano_generation import Task
from piano_generation.generation.generators.base_generator import MidiGenerator


class SeqToSeqIterativeGenerator(MidiGenerator):
    """
    This generation method performs calculations directly on tokens to keep the timing the same at all times.
    """

    def __init__(
        self,
        task: str,
        prompt_context_duration: float,
        target_context_duration: float,
        time_step: float,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ):
        super().__init__(task=task)
        self.prompt_context_duration = prompt_context_duration
        self.target_context_duration = target_context_duration
        self.time_step = time_step
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.task_generator = Task.get_task(task_name=task)

    @staticmethod
    def default_parameters() -> dict:
        return {
            "prompt_context_duration": 15.0,
            "target_context_duration": 10.0,
            "time_step": 5.0,
            "max_new_tokens": 1024,
            "temperature": 1.0,
        }

    @property
    def parameters(self) -> dict:
        return {
            "prompt_context_duration": self.prompt_context_duration,
            "target_context_duration": self.target_context_duration,
            "time_step": self.time_step,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    @staticmethod
    def calculate_token_duration(
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        tokens: list[str],
    ):
        t = 0
        for token in tokens:
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
        return t

    def trim_tokens_front(
        time_step: float,
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        tokens: list[str],
    ):
        t = 0
        tokens = tokens.copy()
        while len(tokens) > 0:
            token = tokens.pop(0)
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
            elif t >= time_step or len(tokens) == 0:
                break

        # in case too much was popped
        if t > time_step:
            dt = t - time_step
            time_tokens = tokenizer.tokenize_time_distance(dt=dt)
            tokens = time_tokens + tokens
        return tokens

    def trim_tokens_back(
        duration: float,
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        tokens: list[str],
    ):
        t = 0
        result = []
        for token in tokens:
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
            result.append(token)
            if t >= duration:
                break
        return result

    def generate(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        device: torch.device,
        additional_tokens: list[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        start_offset = prompt_notes.start.min()
        prompt_notes.start -= start_offset
        prompt_notes.end -= start_offset

        input_tokens = tokenizer.tokenize(prompt_notes)
        step_target_tokens = []
        output_tokens = []
        source_tokens = [self.task_generator.source_token]
        target_tokens = [self.task_generator.target_token]

        if additional_tokens is not None:
            target_tokens += additional_tokens

        for _ in range(self.max_new_tokens):
            step_input_tokens = SeqToSeqTokenwiseGenerator.trim_tokens_back(
                duration=self.prompt_context_duration,
                tokenizer=tokenizer,
                tokens=input_tokens,
            )

            step_input_tokens = source_tokens + step_input_tokens + target_tokens + step_target_tokens

            step_token_ids = [tokenizer.token_to_id[token] for token in step_input_tokens]

            step_token_ids = torch.tensor(
                [step_token_ids],
                device=device,
                dtype=torch.int64,
            )
            next_token = model.generate_new_tokens(
                idx=step_token_ids,
                temperature=self.temperature,
                max_new_tokens=1,
            )

            next_token_id = next_token[0].cpu().numpy()[0]
            next_token = tokenizer.vocab[next_token_id]

            output_tokens.append(next_token)
            step_target_tokens.append(next_token)

            generated_notes_duration = self.calculate_token_duration(
                tokenizer=tokenizer,
                tokens=step_target_tokens,
            )
            # If the generated notes are longer than context_duration, move the generation window time_step to the right
            if generated_notes_duration > self.target_context_duration:
                input_tokens = SeqToSeqTokenwiseGenerator.trim_tokens_front(
                    time_step=self.time_step,
                    tokenizer=tokenizer,
                    tokens=input_tokens,
                )
                step_target_tokens = SeqToSeqTokenwiseGenerator.trim_tokens_front(
                    time_step=self.time_step,
                    tokenizer=tokenizer,
                    tokens=step_target_tokens,
                )

            if (
                SeqToSeqTokenwiseGenerator.calculate_token_duration(
                    tokenizer=tokenizer,
                    tokens=input_tokens,
                )
                == 0
            ):
                break
        target_notes = tokenizer.untokenize(tokens=output_tokens)
        return prompt_notes, target_notes


class SeqToSeqTokenwiseGenerator(MidiGenerator):
    """
    This generation method performs calculations directly on tokens to keep the timing the same at all times.
    """

    def __init__(
        self,
        task: str,
        prompt_context_length: int,
        target_context_length: int,
        time_step: float,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ):
        super().__init__(task=task)
        self.prompt_context_length = prompt_context_length
        self.target_context_length = target_context_length
        self.time_step = time_step
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.task_generator = Task.get_task(task_name=task)

    @staticmethod
    def default_parameters() -> dict:
        return {
            "prompt_context_length": 1024,
            "target_context_length": 512,
            "time_step": 2.0,
            "max_new_tokens": 4096,
            "temperature": 1.0,
        }

    @property
    def parameters(self) -> dict:
        return {
            "prompt_context_length": self.prompt_context_length,
            "target_context_length": self.target_context_length,
            "time_step": self.time_step,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    @staticmethod
    def calculate_token_duration(
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        tokens: list[str],
    ):
        t = 0
        for token in tokens:
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
        return t

    @staticmethod
    def trim_tokens_front(
        time_step: float,
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        tokens: list[str],
    ):
        t = 0
        tokens = tokens.copy()
        while len(tokens) > 0:
            token = tokens.pop(0)
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
            if (t >= time_step and re.search(".T$", token) is None) or len(tokens) == 0:
                break

        # in case too much was popped
        if t > time_step:
            dt = t - time_step
            time_tokens = tokenizer.tokenize_time_distance(dt=dt)
            tokens = time_tokens + tokens
        return tokens

    @staticmethod
    def trim_tokens_back(
        duration: float,
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        tokens: list[str],
    ):
        t = 0
        tokens = tokens.copy()
        result = []
        for token in tokens:
            if re.search(".T$", token) is not None:
                dt: float = tokenizer.token_to_dt[token]
                t += dt
            result.append(token)
            if t >= duration:
                break
        return result

    def generate(
        self,
        prompt_notes: pd.DataFrame,
        model: nn.Module,
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        device: torch.device,
        additional_tokens: list[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        start_offset = prompt_notes.start.min()
        prompt_notes.start -= start_offset
        prompt_notes.end -= start_offset

        input_tokens = tokenizer.tokenize(prompt_notes)
        prompt_notes = tokenizer.untokenize(input_tokens)

        step_target_tokens = []
        output_tokens = []
        source_tokens = [self.task_generator.source_token]
        target_tokens = [self.task_generator.target_token]

        if additional_tokens is not None:
            target_tokens += additional_tokens

        for _ in range(self.max_new_tokens):
            step_input_tokens = input_tokens[: self.prompt_context_length]

            step_input_tokens = source_tokens + step_input_tokens + target_tokens + step_target_tokens

            step_token_ids = [tokenizer.token_to_id[token] for token in step_input_tokens]

            step_token_ids = torch.tensor(
                [step_token_ids],
                device=device,
                dtype=torch.int64,
            )
            next_token = model.generate_new_tokens(
                idx=step_token_ids,
                temperature=self.temperature,
                max_new_tokens=1,
            )

            next_token_id = next_token[0].cpu().numpy()[0]
            next_token = tokenizer.vocab[next_token_id]

            output_tokens.append(next_token)
            step_target_tokens.append(next_token)

            generated_notes_length = len(step_target_tokens)
            # If the generated notes are longer than context_duration, move the generation window time_step to the right
            if generated_notes_length > self.target_context_length:
                input_tokens = SeqToSeqTokenwiseGenerator.trim_tokens_front(
                    time_step=self.time_step,
                    tokenizer=tokenizer,
                    tokens=input_tokens,
                )
                step_target_tokens = SeqToSeqTokenwiseGenerator.trim_tokens_front(
                    time_step=self.time_step,
                    tokenizer=tokenizer,
                    tokens=step_target_tokens,
                )

            if (
                SeqToSeqTokenwiseGenerator.calculate_token_duration(
                    tokenizer=tokenizer,
                    tokens=input_tokens,
                )
                == 0
            ):
                break
        target_notes = tokenizer.untokenize(tokens=output_tokens)
        return prompt_notes, target_notes
