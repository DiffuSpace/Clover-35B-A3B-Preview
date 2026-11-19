"""Single-turn token roles and fixed-length collation; each row is one sample."""

from enum import IntEnum

import torch

SEQUENCE_LENGTH = 4096
IGNORE_INDEX = -100


class Role(IntEnum):
    BATCH_PAD = 0
    PROMPT = 1
    PRE_PAD = 2
    RESPONSE = 3
    POST_PAD = 4


def align_sample(prompt, response, *, block_size, pad_id, length=SEQUENCE_LENGTH):
    """Preserve EOS by rejecting oversized examples instead of truncating them.

    Alignment PAD and batch PAD may share an ID. Roles, rather than token
    equality, determine validity and supervision. Positions skip PRE_PAD.
    """
    if block_size <= 0 or length % block_size:
        raise ValueError("Sequence length must be a positive multiple of block size")
    if not prompt or not response:
        raise ValueError("Both prompt and response must be nonempty")
    pre = -len(prompt) % block_size
    post = -len(response) % block_size
    size = len(prompt) + pre + len(response) + post
    if size > length:
        raise ValueError(f"Block-aligned sample length {size} exceeds fixed length {length}")
    ids = list(prompt) + [pad_id] * pre + list(response) + [pad_id] * post
    roles = ([Role.PROMPT] * len(prompt) + [Role.PRE_PAD] * pre
             + [Role.RESPONSE] * len(response) + [Role.POST_PAD] * post)
    positions = list(range(len(prompt))) + [len(prompt) - 1] * pre
    positions += list(range(len(prompt), len(prompt) + len(response) + post))
    labels = [IGNORE_INDEX] * (len(prompt) + pre) + list(response) + [pad_id] * post
    missing = length - size
    return {
        "input_ids": ids + [pad_id] * missing,
        "labels": labels + [IGNORE_INDEX] * missing,
        "position_ids": positions + [0] * missing,
        "token_roles": [int(r) for r in roles] + [int(Role.BATCH_PAD)] * missing,
    }


def encode_single_turn(tokenizer, messages, *, block_size, system=None, length=SEQUENCE_LENGTH):
    if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
        raise ValueError("Clover SFT accepts exactly one user and one assistant message")
    if not all(isinstance(m["content"], str) and m["content"].strip() for m in messages):
        raise ValueError("Single-turn messages must contain nonempty text")
    context = ([{"role": "system", "content": system}] if system else []) + messages[:1]
    prompt = tokenizer.apply_chat_template(
        context, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False,
    )
    response = tokenizer.encode(messages[1]["content"], add_special_tokens=False)
    response.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The native Clover tokenizer must define PAD and EOS")
    return align_sample(prompt, response, block_size=block_size, pad_id=tokenizer.pad_token_id, length=length)


def collate_samples(samples, *, mask_id, generator=None):
    """Draw one noise rate per sample before forward/recompute.

    Labels contain only sampled denoising targets. This lets the Trainer count
    actual supervised tokens across gradient accumulation and DP ranks.
    """
    batch = {key: torch.tensor([row[key] for row in samples], dtype=torch.long)
             for key in ("input_ids", "labels", "position_ids", "token_roles")}
    roles = batch["token_roles"]
    eligible = (roles == Role.RESPONSE) | (roles == Role.POST_PAD)
    if not eligible.any(-1).all():
        raise ValueError("Every sample must contain response targets")
    selected = torch.zeros_like(eligible)
    # Match the ordinary mask-diffusion empty-draw retry/fallback convention.
    for row in range(len(samples)):
        for _ in range(10):
            rate = 0.001 + 0.999 * torch.rand((), generator=generator)
            selected[row] = (torch.rand(roles.shape[1], generator=generator) < rate) & eligible[row]
            if selected[row].any():
                break
        if not selected[row].any():
            choices = eligible[row].nonzero(as_tuple=True)[0]
            index = torch.randint(choices.numel(), (), generator=generator)
            selected[row, choices[index]] = True
    batch["noisy_input_ids"] = batch["input_ids"].masked_fill(selected, mask_id)
    batch["labels"] = batch["labels"].masked_fill(~selected, IGNORE_INDEX)
    batch["attention_mask"] = ((roles != Role.PRE_PAD) & (roles != Role.BATCH_PAD)).long()
    return batch
