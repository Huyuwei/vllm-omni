# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal GLM-Image image editing (img2img) example for vLLM-Omni multistage.

Usage:
    python minimal_edit.py \
        --model-path zai-org/GLM-Image \
        --image input.png \
        --prompt "Turn this into a watercolor painting" \
        --output output_glm_edit.png
"""

import argparse

from PIL import Image
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

GLM_IMAGE_EOS_TOKEN_ID = 16385
GLM_IMAGE_VISION_VOCAB_SIZE = 16512
DEFAULT_STAGE_CONFIG_PATH = "vllm_omni/model_executor/stage_configs/glm_image.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal GLM-Image edit example")
    parser.add_argument("--model-path", required=True, help="HF model id or local model path")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", required=True, help="Edit instruction")
    parser.add_argument("--output", default="output_glm_edit.png", help="Output image path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--stage-config", default=DEFAULT_STAGE_CONFIG_PATH)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    prompt = {
        "prompt": args.prompt,
        "multi_modal_data": {"image": image},
        "height": image.height,
        "width": image.width,
        "mm_processor_kwargs": {
            "target_h": image.height,
            "target_w": image.width,
        },
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
    }

    ar_sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.75,
        top_k=GLM_IMAGE_VISION_VOCAB_SIZE,
        max_tokens=4096,
        stop_token_ids=[GLM_IMAGE_EOS_TOKEN_ID],
        seed=args.seed,
        detokenize=False,
    )
    diffusion_sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=image.height,
        width=image.width,
        seed=args.seed,
    )

    omni = Omni(model=args.model_path, stage_configs_path=args.stage_config)
    saved = False
    for stage_outputs in omni.generate(
        [prompt], [ar_sampling_params, diffusion_sampling_params], py_generator=True
    ):
        if stage_outputs.final_output_type != "image":
            continue

        output = stage_outputs.request_output[0]
        images = output.images if hasattr(output, "images") else []
        if not images and hasattr(output, "multimodal_output"):
            images = output.multimodal_output.get("images", [])

        if images:
            images[0].save(args.output)
            saved = True
            break

    omni.close()

    if not saved:
        raise RuntimeError("No image output found from GLM-Image edit request")

    print(f"Saved edited image to {args.output}")


if __name__ == "__main__":
    main()
