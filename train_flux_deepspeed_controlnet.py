import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from PIL import Image, ImageOps
import accelerate
import datasets
from datasets import load_dataset
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torchvision import transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_controlnet, load_t5)
from controlnet_aux import LineartDetector
from image_datasets.sketch_dataset import loader

MAX_CONSISTENCY = 0.14


if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")

def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config


def get_train_dataset(train_data_dir, accelerator):

    data_files = {}
    data_files["train"] = os.path.join(train_data_dir, "**")
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        # cache_dir=args.cache_dir,
    )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # column_names = dataset["train"].column_names

    # # 6. Get the column names for input/target.
    # if args.image_column is None:
    #     image_column = column_names[0]
    #     logger.info(f"image column defaulting to {image_column}")
    # else:
    #     image_column = args.image_column
    #     if image_column not in column_names:
    #         raise ValueError(
    #             f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
    #         )

    # if args.caption_column is None:
    #     caption_column = column_names[1]
    #     logger.info(f"caption column defaulting to {caption_column}")
    # else:
    #     caption_column = args.caption_column
    #     if caption_column not in column_names:
    #         raise ValueError(
    #             f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
    #         )

    # with accelerator.main_process_first():
    #     train_dataset = dataset["train"].shuffle(seed=args.seed)
    #     if args.max_train_samples is not None:
    #         train_dataset = train_dataset.select(range(args.max_train_samples))
    return dataset["train"]

def prepare_train_dataset(dataset, accelerator, args):
    image_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.img_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.img_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.img_size),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[args.image_column]]
        images = [image_transforms(image) for image in images]

        # just passing the ground truth as conditioning image. It will be converted online during training
        # so that each GPU can have it's own loader
        conditioning_images = [
            image.convert("RGB") for image in examples[args.image_column]
        ]
        conditioning_images = [
            conditioning_image_transforms(image) for image in conditioning_images
        ]

        examples["pixel_values"] = images
        examples["conditioning_images"] = conditioning_images

        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = [example["conditioning_images"] for example in examples]

    prompts = [example["text"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "conditioning_images": conditioning_pixel_values,
        "prompts": prompts
    }


def detect_sketch(img, detector):
    # bunch of augmentation to make the model more tolerant to different styles of sketches
    COARSE_PROB = 0.2
    detect_res = [380, 512, 768, 1024]
    brightness_range = [0.6, 1.0]
    detect_cycles = [1, 2]

    c_coarse = random.uniform(0, 1) < COARSE_PROB
    c_res = random.choice(detect_res)
    c_brightness = random.uniform(brightness_range[0], brightness_range[1])
    # Don't run the lineart detector multiple time with a coarse detection as the results will be terrible
    c_cycles = 1 if c_coarse else random.choice(detect_cycles)

    feed = img
    for _ in range(c_cycles):
        out = detector(
            feed, coarse=c_coarse, detect_resolution=c_res, image_resolution=512
        )
        feed = ImageOps.invert(out)
    out = out.resize(img.size)

    out = np.array(out.convert("L"))
    out = out * c_brightness
    out = Image.fromarray(out).convert("RGB")
    
    ref = ImageOps.invert(img.convert('L'))

    gen = np.array(out.convert('L'))/255.
    ref = np.array(ref)/255.
    overlap = ref*gen
    consistency_score = overlap.sum() / ref.sum()
    if consistency_score > MAX_CONSISTENCY:
        # High consistency between reference image and generated sketch indicate that the reference image was a lineart or a sketch to begin with
        # training on this kind of data will lead to bad quality generations when passing actual sketches to the model
        # When that happens (about 5-7% of the time with MAX_CONSISTENCY=0.15 on MJ data) we instead return a full black image
        out = Image.new('RGB', img.size, (0, 0, 0))

    return out



def get_conditioning_image(img, detector):
    toTensor = transforms.ToTensor()
    ground_truth = img.convert("RGB")
    sketch_img = detect_sketch(ground_truth, detector)
    return toTensor(sketch_img)


def process_batch(batch, accelerator, lineart_detector):
    conditioning_images = [
        get_conditioning_image(img, lineart_detector)
        for img in batch["conditioning_images"]
    ]
    conditioning_pixel_values = torch.stack(conditioning_images)
    conditioning_pixel_values = (
        conditioning_pixel_values.to(memory_format=torch.contiguous_format)
        .float()
        .to(accelerator.device)
    )
    batch["conditioning_pixel_values"] = conditioning_pixel_values

    # batch = compute_embeddings(
    #     batch,
    #     accelerator,
    #     args.proportion_empty_prompts,
    #     text_encoders,
    #     tokenizers,
    #     is_train=True,
    # )


    return batch


def main():
    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    if is_schnell:
        print('Training on top of Schnell, is this supported?')
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    print("DEVICE", accelerator.device)
    dit, vae, t5, clip = get_models(name=args.model_name, device=accelerator.device, offload=False, is_schnell=is_schnell)

    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit.requires_grad_(False)
    dit.to(accelerator.device)

    controlnet = load_controlnet(name=args.model_name, device=accelerator.device, transformer=dit)
    controlnet = controlnet.to(torch.float32)
    controlnet.train()

    optimizer_cls = torch.optim.AdamW

    print(sum([p.numel() for p in controlnet.parameters() if p.requires_grad]) / 1000000, 'parameters')
    optimizer = optimizer_cls(
        [p for p in controlnet.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators").to('cuda')

    # data_files = {}
    # data_files["train"] = os.path.join(args.data_config.img_dir, "**")
    # dataset = load_dataset(
    #     "imagefolder",
    #     data_files=data_files,
    #     # cache_dir=args.cache_dir,
    # )
    # train_dataloader = loader(**args.data_config)
    

    train_dataset = get_train_dataset(args.data_config.img_dir, accelerator)
    train_dataset = prepare_train_dataset(train_dataset, accelerator, args.data_config)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.data_config.num_workers,
    )

    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    controlnet, optimizer, _, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = list(torch.linspace(1, 0, 1000).numpy())
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # img, control_image, prompts = batch
                batch = process_batch(batch, accelerator, lineart_detector)
                
                prompts = batch['prompts']
                img = batch['pixel_values']
                control_image = batch['conditioning_pixel_values']

                # img, prompts = batch
                # control_image = lineart_detector(img)
                # img = torch.from_numpy((np.array(img) / 127.5) - 1)
                # img = img.permute(2, 0, 1)
                # control_image = torch.from_numpy((np.array(control_image) / 127.5) - 1)
                # control_image = control_image.permute(2, 0, 1)

                # control_image = control_image.to(accelerator.device)
                with torch.no_grad():
                    x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))
                    inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)

                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                bs = img.shape[0]
                t = torch.sigmoid(torch.randn((bs,), device=accelerator.device))

                x_0 = torch.randn_like(x_1).to(accelerator.device)
                x_t = (1 - t.unsqueeze(1).unsqueeze(2).repeat(1, x_1.shape[1], x_1.shape[2])) * x_1 + t.unsqueeze(1).unsqueeze(2).repeat(1, x_1.shape[1], x_1.shape[2]) * x_0
                bsz = x_1.shape[0]
                guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)

                block_res_samples = controlnet(
                    img=x_t.to(weight_dtype),
                    img_ids=inp['img_ids'].to(weight_dtype),
                    controlnet_cond=control_image.to(weight_dtype),
                    txt=inp['txt'].to(weight_dtype),
                    txt_ids=inp['txt_ids'].to(weight_dtype),
                    y=inp['vec'].to(weight_dtype),
                    timesteps=t.to(weight_dtype),
                    guidance=guidance_vec.to(weight_dtype),
                )
                # Predict the noise residual and compute loss
                model_pred = dit(
                    img=x_t.to(weight_dtype),
                    img_ids=inp['img_ids'].to(weight_dtype),
                    txt=inp['txt'].to(weight_dtype),
                    txt_ids=inp['txt_ids'].to(weight_dtype),
                    block_controlnet_hidden_states=[
                        sample.to(dtype=weight_dtype) for sample in block_res_samples
                    ],
                    y=inp['vec'].to(weight_dtype),
                    timesteps=t.to(weight_dtype),
                    guidance=guidance_vec.to(weight_dtype),
                )

                loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    #if not os.path.exists(save_path):
                    #        os.mkdir(save_path)

                    accelerator.save_state(save_path)
                    unwrapped_model = accelerator.unwrap_model(controlnet)

                    torch.save(unwrapped_model.state_dict(), os.path.join(save_path, 'controlnet.bin'))
                    logger.info(f"Saved state to {save_path}")


            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
