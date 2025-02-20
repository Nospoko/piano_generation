import torch
from omegaconf import OmegaConf, DictConfig
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from piano_generation import GPT
from piano_generation.artifacts import special_tokens


def load_cfg(checkpoint: dict) -> DictConfig:
    train_config = checkpoint["run_config"]
    return OmegaConf.create(train_config)


def load_checkpoint(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def load_tokenizer(
    cfg: DictConfig,
    special_tokens: list[str],
):
    tokenizer_options = OmegaConf.to_container(cfg.tokenizer)
    tokenizer_config = tokenizer_options["config"]
    if tokenizer_options["class_name"] == "ExponentialTimeTokenizer":
        tokenizer = ExponentialTimeTokenizer.build_tokenizer(tokenizer_config=tokenizer_config)
        tokenizer.add_special_tokens(special_tokens=special_tokens)
        return tokenizer
    else:
        raise NotImplementedError(f"Unknown class name: {tokenizer_options.class_name}")


def initialize_gpt_model(
    cfg: DictConfig,
    checkpoint: dict,
    device: torch.device,
) -> GPT:
    """
    Initializes the GPT model using the given configurations and checkpoint.

    Parameters:
        cfg (DictConfig): The configuration object.
        dataset_config (dict): The dataset configuration.
        checkpoint (dict): The model checkpoint.
        device (torch.device): The device to load the model on.

    Returns:
        GPT: The initialized GPT model.
    """
    tokenizer = ExponentialTimeTokenizer.from_dict(checkpoint["tokenizer_desc"])

    model = GPT(
        config=cfg.model,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=tokenizer.vocab_size,
    )
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model
