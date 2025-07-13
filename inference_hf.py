import argparse
from typing import Any, List, Optional
import os
from pydantic import BaseModel, Field
from platform import system
import shutil
import io

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import numpy as np

from multiprocessing import set_start_method, Queue, Process, Pipe, current_process

from torch import memory_format

try:
    set_start_method('spawn')
except RuntimeError:
    pass

import torch
import gc


if system() == "Windows":
    MAX_RAND = 2**16 - 1
else:
    MAX_RAND = 2**32 - 1

# Configuration
# This two vars will be changed in __main__ function
NUM_GPUS = 1  # Number of simulated GPUs
MAX_GPU_MEMORY = 1


TASK_QUEUE = Queue()
GPU_PROCESSES: List[Process] = []
STYLES_FOLDER = "/lora_styles"
OFFLOAD_EMBD = True
OFFLOAD_VAE = True
TURN_OFF = False



class LoraStyle(BaseModel):
    path: str
    scale: float = Field(default=1.0)
    name: Optional[str] = None


class GenerateArgs(BaseModel):
    """Web request data format"""
    prompt: str
    width: Optional[int] = Field(default=1024)
    height: Optional[int] = Field(default=720)
    num_steps: Optional[int] = Field(default=28)
    guidance: Optional[float] = Field(default=3.5)
    seed: Optional[int] = Field(default_factory=lambda: np.random.randint(0, MAX_RAND), gt=0, lt=MAX_RAND)
    lora_personal: Optional[LoraStyle] = None
    lora_styles: Optional[list[LoraStyle]] = None



app = FastAPI()



def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Flux API server")
    # parser.add_argument("-c", "--config-path", type=str, help="Path to the configuration file, if not provided, the model will be loaded from the command line arguments")
    parser.add_argument("-p", "--port", type=int, default=8088, help="Port to run the server on")
    parser.add_argument("-H", "--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--no_offload_embd", action="store_true", help="Keep embedding model always on GPU")
    parser.add_argument("--no_offload_vae", action="store_true", help="Keep VAE model always on GPU")
    parser.add_argument("--no_quant", action="store_true", help="Do not quantize Flux model")

    return parser.parse_args()


def process_data_on_gpu(encoder, model, vae, device, vars: dict[str, Any], pipe_in: Pipe):
    """Image generating steps with lora loads and memory management"""
    # encoder
    print(f"GPU {device} encoder: {vars['prompt'][:40]}")

    if OFFLOAD_EMBD:
        encoder.to("cuda")
    with torch.inference_mode():
        prompt_embeds, pooled_prompt_embeds, text_ids = encoder.encode_prompt(
            vars['prompt'], prompt_2=None, max_sequence_length=512, num_images_per_prompt=1)
    if OFFLOAD_EMBD:
        encoder.to('cpu')
        flush()

    # loading loras
    print(f"GPU {device} loading loras")
    lora_names = []
    lora_scales = []
    if not vars.get('lora_personal', None) is None:
        model.load_lora_weights(vars['lora_personal']['path'], adapter_name='user')
        lora_names.append("user")
        lora_scales.append(1.0)
        del(vars['lora_personal'])
    if not vars.get('lora_styles', None) is None:
        for style in vars['lora_styles']:
            if len(style['path']) == 0:
                continue
            model.load_lora_weights(style['path'], adapter_name=style['name'])
            lora_names.append(style['name'])
            lora_scales.append(style['scale'])
        del(vars['lora_styles'])

    if len(lora_names) > 0:
        model.set_adapters(lora_names, adapter_weights=lora_scales)

    vars['num_inference_steps'] = vars['num_steps']
    vars['guidance_scale'] = vars['guidance']
    del(vars['guidance'], vars['num_steps'], vars['seed'])

    # transformer
    print(f"GPU {device} transformer: {vars['prompt'][:40]}")
    with torch.inference_mode():
        latents = model(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                        num_inference_steps=vars['num_inference_steps'],
                        guidance_scale=vars['guidance_scale'],
                        height=vars['height'],
                        width=vars['width'],
                        output_type="latent").images

    if len(lora_names) > 0:
        model.unload_lora_weights(reset_to_overwritten_params=True)
    flush()

    # vae
    from diffusers.image_processor import VaeImageProcessor
    from diffusers import FluxPipeline
    print(f"GPU {device} vae: {vars['prompt'][:40]}")

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    if OFFLOAD_VAE:
        vae.to("cuda")
    with torch.inference_mode():
        latents = FluxPipeline._unpack_latents(latents, vars['height'], vars['width'], vae_scale_factor)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="pil")[0]
    if OFFLOAD_VAE:
        vae.to('cpu')

    if pipe_in is None: # warmup
        return

    # bytes sending
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='jpeg')
    imgByteArr = imgByteArr.getvalue()
    pipe_in.send(imgByteArr)
    pipe_in.send(None)  # Signal completion





def gpu_worker(gpu_id: int, world_size, task_queue, args):
    """Loads models and listening new tasks from Queue"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = "/workspace/hf"
    os.environ["HF_HUB_CACHE"] = "/workspace/hf"

    import torch
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 20
    torch.set_float32_matmul_precision("high")


    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from diffusers import FluxTransformer2DModel, FluxPipeline, AutoencoderKL

    from para_attn.parallel_vae.diffusers_adapters import parallelize_vae
    from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_transformer

    change_globals(args, max_mem=True)
    print(f"GPU {gpu_id} globals: {OFFLOAD_EMBD = }  {OFFLOAD_VAE = }")

    max_memory = {k: "0GB" for k in range(world_size)}
    max_memory[gpu_id] = str(MAX_GPU_MEMORY) + "GB"
    max_memory["cpu"] = "0GB"

    print(f"GPU {gpu_id} loading model. {max_memory = }")
    dtype = torch.float16 if not args.no_quant else torch.bfloat16

    # start model on GPU
    encoder = FluxPipeline.from_pretrained("black-forest-labs/flux.1-dev", transformer=None, vae=None,
                                           torch_dtype=dtype, max_memory=max_memory)

    if OFFLOAD_EMBD:
        encoder.to('cpu')
    else:
        encoder.to('cuda')

    transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/flux.1-dev", subfolder="transformer",
                max_memory=max_memory,
                quantization_config=DiffusersBitsAndBytesConfig(load_in_8bit=True) if not args.no_quant else None,
                torch_dtype=dtype,)

    model = FluxPipeline.from_pretrained("black-forest-labs/flux.1-dev", transformer=transformer, text_encoder=None,
                                         text_encoder_2=None, tokenizer=None, tokenizer_2=None, vae=None,
                                         torch_dtype=dtype).to("cuda")
    apply_cache_on_transformer(model, residual_diff_threshold=0.12)
    model.to(memory_format=torch.channels_last)

    vae = AutoencoderKL.from_pretrained("black-forest-labs/flux.1-dev", subfolder="vae",
                                        torch_dtype=dtype, max_memory=max_memory)
    parallelize_vae(vae)
    vae.to(memory_format=torch.channels_last)

    if OFFLOAD_VAE:
        vae.to('cpu')
    else:
        vae.to('cuda')

    # compile
    # torch.compile configuration
    config = torch._inductor.config
    config.conv_1x1_as_mm = True
    # with torch.inference_mode():
        # for block in model.transformer.transformer_blocks:
        #     block.compile()
        # for block in model.transformer.single_transformer_blocks:
        #     block.compile()

        # for extra in ["context_embedder", "x_embedder", "time_text_embed", "proj_out", "pos_embed",]:
        #     getattr(model.transformer, extra).compile()
        # encoder.text_encoder_2.compile()
        # encoder.text_encoder.compile()
    vae.fuse_qkv_projections()
    vae = torch.compile(vae)

    # warmup
    print(f"GPU {gpu_id} warmup.")
    warmup_dict = dict(height=1024, width=1024, num_steps=4, guidance=3.5, seed=10,
        prompt="A beautiful Warmup image generation used nn.Linear input scales prior to compilation 😉",)
    process_data_on_gpu(encoder, model, vae, gpu_id, warmup_dict, None)


    # generation loop
    print(f"GPU {gpu_id} worker processes started.")
    while True:
        task = task_queue.get()
        if task is None:  # Shutdown signal
            return 'closing'
        vars, pipe_in = task
        # child_conn = pickle.loads(child_conn)
        print(f"GPU worker {current_process().name} received task: {vars['prompt'][:40]}")
        process_data_on_gpu(encoder, model, vae, gpu_id, vars, pipe_in)


def keep_worker_alive(gpu_id: int, world_size, task_queue, args):
    """Keeps each worker alive"""
    while True:
        try:
            result = gpu_worker(gpu_id, world_size, task_queue, args)
            if result == 'closing':
                print(f"GPU-{gpu_id} finished")
                return
        except Exception as e:
            print(f"GPU-{gpu_id} got error", e)
            print(f"Restarting worker GPU-{gpu_id}")


def output_generator(pipe_out: Pipe):
    """
    Generator function to yield output from the GPU worker via the pipe.
    """
    while True:
        try:
            output = pipe_out.recv()
            if output is None:
                # print("output_generator: None, closing")
                pipe_out.close()
                break
            yield output
        except EOFError:
            # print("output_generator: EOF, closing")
            pipe_out.close()
            break


@app.post("/generate")
async def generate(args: GenerateArgs):
    """
    Args:
        args (GenerateArgs):

    Returns:
        StreamingResponse: The generated image as streaming jpeg bytes.
    Example:
        data = {"prompt": "Headshot of a handsome young U5ER: Dark green sweater with buttons and shawl collar, black hair, short beard. Serious expression on a black background, soft studio lighting.",
                "width": 1024, "height": 1024, "num_steps": 50,
                "lora_styles": [
                    {"path": "/workspace/lora_styles/amateurphoto-v6-forcu.safetensors", "scale": 0.7, "name": "amateur"},
                    {"path": "/workspace/lora_styles/Disney-Studios-Flux-000008.safetensors", "scale": 0.2, "name": "Disney"},
                ],
                "lora_personal": {"path": "/workspace/lora_styles/h100-bs2-26.safetensors", "scale": 1},
                }

        res = requests.post("http://localhost:8088/generate", json=data, stream=True)
    """

    # downloading if need lora styles
    if not args.lora_styles is None:
        for i in range(len(args.lora_styles)):
            path = args.lora_styles[i].path
            if not os.path.exists(path):
                args.lora_styles[i].path = ""
                continue
            basename = os.path.basename(path)
            local_path = os.path.join(STYLES_FOLDER, basename)
            if not os.path.exists(local_path):  # download and load from local
                shutil.copy(path, local_path)

            args.lora_styles[i].path = local_path

            if args.lora_styles[i].name is None:
                args.lora_styles[i].name = basename.split(".safet")[0]


    vars = args.model_dump()
    pipe_out, pipe_in = Pipe(duplex=False)
    TASK_QUEUE.put((vars, pipe_in))  # Put data and the child pipe end in the queue
    # return StreamingResponse(output_generator(pipe_out), media_type="text/plain")
    return StreamingResponse(output_generator(pipe_out), media_type="image/jpeg")


def change_globals(args, num_gpus=False, max_mem=False):
    global OFFLOAD_EMBD, OFFLOAD_VAE
    if args.no_offload_embd:
        OFFLOAD_EMBD = False
        # print("OFFLOAD_EMBD changed to False")
    if args.no_offload_vae:
        OFFLOAD_VAE = False
        # print("OFFLOAD_VAE changed to False")
    if num_gpus:
        global NUM_GPUS
        NUM_GPUS = torch.cuda.device_count()
    if max_mem:
        global MAX_GPU_MEMORY
        MAX_GPU_MEMORY = int(torch.cuda.mem_get_info(0)[1] / 1024 ** 2 / 1000)



def shutdown_gpus():
    """
    Signals the GPU worker processes to shut down when the FastAPI app stops.
    """
    global TURN_OFF
    TURN_OFF = True
    for _ in range(NUM_GPUS):
        TASK_QUEUE.put(None)
    for process in GPU_PROCESSES:
        process.join()
    print("GPU worker processes stopped.")


app.add_event_handler("shutdown", shutdown_gpus)


if __name__ == "__main__":

    args = parse_args()
    max_gpu_memory = int(torch.cuda.mem_get_info(0)[1] / 1024 ** 2 / 1000)
    if max_gpu_memory > 37:
        args.no_offload_embd = True
        args.no_offload_vae = True
    change_globals(args, True, True)
    print(f"\nMain globals:\n\t{NUM_GPUS = }\n\t{MAX_GPU_MEMORY = }\n\t{OFFLOAD_EMBD = }\n\t{OFFLOAD_VAE = }\n")

    os.makedirs(STYLES_FOLDER, exist_ok=True)

    for i in range(NUM_GPUS):
        process = Process(target=keep_worker_alive, args=(i, NUM_GPUS, TASK_QUEUE, args), name=f"GPU-{i}")
        GPU_PROCESSES.append(process)
        process.start()
    print("GPU worker processes launched.")


    uvicorn.run(app, host=args.host, port=args.port)
