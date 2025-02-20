import torch
import numpy as np
import pandas as pd
from torch import nn
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from piano_generation import Task
from piano_generation.generation.generators.base_generator import MidiGenerator


class NoteToNoteGenerator(MidiGenerator):
    """
    This generation method assures equal number of notes in prompt and target contexts at all times
    during generation. It is ideal for tasks such as velocity denoising or performance task -
    where there should be one note generated per one note in the prompt.
    """

    def __init__(
        self,
        task: str,
        prompt_context_notes: int,
        target_context_notes: int,
        step: int,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ):
        super().__init__(task=task)
        self.prompt_context_notes = prompt_context_notes
        self.target_context_notes = target_context_notes
        self.step = step
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.task_generator = Task.get_task(task_name=task)

    @staticmethod
    def default_parameters() -> dict:
        return {
            "prompt_context_notes": 128,
            "target_context_notes": 64,
            "step": 16,
            "max_new_tokens": 4096,
            "temperature": 1.0,
        }

    @property
    def parameters(self) -> dict:
        return {
            "prompt_context_notes": self.prompt_context_notes,
            "target_context_notes": self.target_context_notes,
            "step": self.step,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    @staticmethod
    def calculate_notes_in_tokens(
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        tokens: list[str],
    ):
        try:
            notes = tokenizer.untokenize(
                tokens=tokens,
                complete_notes=False,
            )
        except KeyError:
            # KeyError in tokenizer.untokenizes is raised when there are no full notes in tokens.
            return 0
        return len(notes)

    @staticmethod
    def trim_notes_front(
        step: int,
        tokens: list[str],
    ):
        n_notes = 0
        it = 0
        for token in tokens:
            if token.startswith("NOTE_OFF"):
                n_notes += 1
            if n_notes >= step:
                break
            it += 1

        return tokens[it:]

    @staticmethod
    def trim_notes_back(
        size: float,
        tokens: list[str],
    ):
        n_notes = 0
        trimmed_tokens = np.array([], dtype=str)
        for token in tokens:
            if token.startswith("NOTE_OFF"):
                n_notes += 1
            trimmed_tokens = np.append(arr=trimmed_tokens, values=[token])
            if n_notes >= size:
                break

        return trimmed_tokens.tolist()

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
            step_input_tokens = self.trim_notes_back(
                size=self.prompt_context_notes,
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
            generated_notes_size = self.calculate_notes_in_tokens(
                tokenizer=tokenizer,
                tokens=step_target_tokens,
            )
            # If the generated notes are longer than context_duration, move the generation window to the right
            if generated_notes_size > self.target_context_notes:
                input_tokens = NoteToNoteGenerator.trim_notes_front(
                    step=self.step,
                    tokens=input_tokens,
                )
                step_target_tokens = NoteToNoteGenerator.trim_notes_front(
                    step=self.step,
                    tokens=step_target_tokens,
                )
                if (
                    NoteToNoteGenerator.calculate_notes_in_tokens(
                        tokenizer=tokenizer,
                        tokens=input_tokens,
                    )
                    == 0
                ):
                    break

        target_notes = tokenizer.untokenize(
            tokens=output_tokens,
            complete_notes=False,
        )
        return prompt_notes, target_notes
