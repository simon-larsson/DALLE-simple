import os
import random
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.training.common_utils import shard_prng_key, shard
from flax.jax_utils import replicate
from PIL import Image
from tqdm import trange
import wandb

os.environ[
    "XLA_PYTHON_CLIENT_ALLOCATOR"
] = "platform"  # https://github.com/saharmor/dalle-playground/issues/14#issuecomment-1147849318
os.environ["WANDB_SILENT"] = "true"
wandb.init(anonymous="must")


def create_p_generate(model):
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    return p_generate


def create_p_decode(model):
    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return model.decode_code(indices, params=params)

    return p_decode


class DalleMini:
    def __init__(
        self,
        dalle_version="dalle-mini/dalle-mini/mega-1-fp16:latest",
        dalle_commit_id=None,
        vqgan_version="dalle-mini/vqgan_imagenet_f16_16384",
        vqgan_commit_id="e93a26e7707683d349bf5d5c41c5b0ef69b677a9",
        clip_version="openai/clip-vit-base-patch32",
        clip_commit_id=None,
        seed=None,
    ):
        # load dalle
        self.dalle, dalle_params = DalleBart.from_pretrained(
            dalle_version, revision=dalle_commit_id, dtype=jnp.float16, _do_init=False
        )
        self.dalle_processor = DalleBartProcessor.from_pretrained(dalle_version, revision=dalle_commit_id)
        self.dalle_params = replicate(dalle_params)

        # load vqgan
        self.vqgan, vqgan_params = VQModel.from_pretrained(vqgan_version, revision=vqgan_commit_id, _do_init=False)
        self.vqgan_params = replicate(vqgan_params)

        # load clip
        self.clip, clip_params = FlaxCLIPModel.from_pretrained(
            clip_version, revision=clip_commit_id, dtype=jnp.float16, _do_init=False
        )
        self.clip_processor = CLIPProcessor.from_pretrained(clip_version, revision=clip_commit_id)
        self.clip_params = replicate(clip_params)

        # create vectorized jax functions with initialized models
        self._p_generate = create_p_generate(self.dalle)
        self._p_decode = create_p_decode(self.vqgan)

        # set seed
        self.seed = seed

    def generate_from_prompt(
        self,
        prompt,
        n_predictions,
        gen_top_k=None,
        gen_top_p=None,
        temperature=None,
        cond_scale=10.0,
    ):
        # (see https://huggingface.co/blog/how-to-generate)

        if self.seed is None:
            seed = random.randint(0, 2**32 - 1)
            key = jax.random.PRNGKey(seed)
        else:
            key = jax.random.PRNGKey(self.seed)

        tokenized_prompt = self.dalle_processor([prompt])
        tokenized_prompt = replicate(tokenized_prompt)

        images = []
        for i in trange(max(n_predictions // jax.device_count(), 1)):
            # get a new key
            key, subkey = jax.random.split(key)
            # generate images
            encoded_images = self._p_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                self.dalle_params,
                gen_top_k,
                gen_top_p,
                temperature,
                cond_scale,
            )
            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            # decode images
            decoded_images = self._p_decode(encoded_images, self.vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)

        return images
