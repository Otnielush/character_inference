import oyaml as yaml
import os, sys
from PIL import Image, ImageFont, ImageDraw
from loguru import logger
from tqdm.auto import tqdm
from argparse import ArgumentParser


Font = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
if os.name == 'nt':
    Font = "arial.ttf"


def concat_images(img_list, width, height):
    "img_list: 2 dim array, 0 dim - images in rows, 1 dim - images in columns"
    cols = max([len(i) for i in img_list])
    rows = len(img_list)
    width += 6
    height += 6

    composite = Image.new('RGB', (width * cols - 6, height * rows - 6))
    for r in range(rows):
        for c in range(len(img_list[r])):
            composite.paste(Image.open(img_list[r][c]), (width * c, height * r))
    return composite

def add_titles_concat_images(titles, concat_images, width, height):
    width += 6
    height += 6
    side = Image.new('RGB', (700 + concat_images.width, height * len(titles) - 6))
    draw = ImageDraw.Draw(side)
    font = ImageFont.truetype(Font, 80)

    for i, title in enumerate(titles):
        draw.text((5, height * i + height / 2), title, (255,255,255), font=font)
    side.paste(concat_images, (700, 0))
    return side

def matrix_titles_images(row_titles, col_titles, concat_images, width, height, row_name="", col_name=""):
    width += 6
    height += 6
    side = Image.new('RGB', (500 + concat_images.width, 100 + concat_images.height))
    draw = ImageDraw.Draw(side)
    font = ImageFont.truetype(Font, 70, encoding="unic")
    draw.text((5, 200), str(row_name) + ":", (255,255,255), font=font)
    draw.text((300, 15), str(col_name) + ":", (255,255,255), font=font)
    for i, title in enumerate(row_titles):
        draw.text((10, height * i + height / 2 + 100), str(title), (255,255,255), font=font)
    for i, title in enumerate(col_titles):
        draw.text((width * i + width / 2 + 500, 15), str(title), (255,255,255), font=font)
    side.paste(concat_images, (500, 100))
    return side



def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("training_config", help="path to training config to get parameters")
    parser.add_argument("--mode", type=str, default="1", choices=['1', '2'],
                        help="""1 - Matrix of 1 prompt with different scales of 2 lora adapters. 
                        First prompt from config; two first Lora adapters for effects; Additional lora adapters from permanent_adapters in config.
                        2 - Matrix of different prompts and Lora adapters. 
                        Rows - lora adapters (adapter_paths) from config. Columns - prompts from config. Additional lora adapters from permanent_adapters in config""")

    return parser.parse_args()




if __name__ == "__main__":
    args = _parse_args()

    with open(args.training_config, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader).get('config', {})

    matrix_config = config['process'][0].get('matrix_params', {})
    if not matrix_config.get('generate_matrix', False):
        exit()


    from flux_pipeline import FluxPipeline

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    logger.remove(0)
    logger.add(sys.stderr, level="ERROR")


    # -------------- Matrix generation
    print('Loading model')
    pipe = FluxPipeline.load_pipeline_from_config_path(os.path.join(os.path.dirname(__file__), "configs/config-dev-H100.json"))
    # pipe = FluxPipeline.load_pipeline_from_config_path("configs/config-dev-offload-3-4090.json")
    # character Lora
    lora_path = os.path.join(config['process'][0]['training_folder'], config['name'], config['name'] + '.safetensors')
    trigger = config['process'][0].get('trigger_word', "")
    pipe.model.load_lora(lora_path, 1.0, "User")

    # additional permanent Lora effects
    additional_triggers = []
    permanent_adapter_paths = matrix_config.get('permanent_adapter_paths', [])
    permanent_adapter_paths = permanent_adapter_paths if isinstance(permanent_adapter_paths, list) else []
    for additional_lora in permanent_adapter_paths:
        if not os.path.exists(additional_lora):
            continue
        name = os.path.basename(additional_lora).split(('.safet'))[0]
        adap_trigger_path = os.path.join(os.path.dirname(additional_lora), name + '.txt')
        adap_trigger = ""
        if os.path.exists(adap_trigger_path):
            with open(adap_trigger_path, 'r') as f:
                adap_trigger = f.read()
        additional_triggers.append(adap_trigger)
        pipe.model.load_lora(additional_lora, scale=0.4)
    additional_triggers = ', '.join(additional_triggers) if len(additional_triggers) > 0 else ""

    width = matrix_config.get('width', 1024)
    height = matrix_config.get('height', 1024)
    prompts = matrix_config.get('prompts', [])
    prompts = [p.replace('[trigger]', trigger) for p in prompts]

    gen_images = []

    if args.mode == "1":
        # Get and load 2 first loras from config
        lora_adapters = []
        i = len(matrix_config.get('adapter_paths', []))
        while len(lora_adapters) < 2:
            i -= 1
            if i <= -1:
                break
            path = matrix_config['adapter_paths'][i]
            if not os.path.exists(path):
                continue
            name = os.path.basename(path).split(('.safet'))[0]
            adap_trigger_path = os.path.join(os.path.dirname(path), name + '.txt')
            adap_trigger = ""
            if os.path.exists(adap_trigger_path):
                with open(adap_trigger_path, 'r') as f:
                    adap_trigger = f.read()

            lora_adapters.append({'name': name, "trigger": adap_trigger, "path": path, 'scale': 1.0})

        prompt_ = (additional_triggers + " " + prompts[0]).strip()
        for adapter in tqdm(lora_adapters):
            prompt_ = (adapter['trigger'] + ' ' + prompt_).strip()
            pipe.load_lora(lora_path=adapter['path'], scale=adapter['scale'], name=adapter['name'])


        scales1 = [0.8, 0.9, 1.0, 1.1, 1.2]
        scales2 = [0.8, 0.9, 1.0, 1.1, 1.2]

        pbar = tqdm(total=len(scales1) * len(scales2))
        for s1 in scales1:
            s1_images = []
            for s2 in scales2:
                lora_adapters[0]['scale'] = s1
                lora_adapters[1]['scale'] = s2
                for adapter in lora_adapters:
                    pipe.load_lora(lora_path=adapter['path'], scale=adapter['scale'], name=adapter['name'], silent=True)
                output_jpeg_bytes = pipe.generate(prompt=prompt_, width=width, height=height,
                                        num_steps=matrix_config['sample_steps'], guidance=4, seed=None, silent=True)
                s1_images.append(output_jpeg_bytes)
                pbar.update(1)
            gen_images.append(s1_images)
        pbar.close()

        ci = concat_images(gen_images, width, height)
        tci = matrix_titles_images(row_titles=scales1, col_titles=scales2, concat_images=ci, width=width,
                                   height=height, row_name=lora_adapters[0]['name'], col_name=lora_adapters[1]['name'])

    # second type of matrix
    elif args.mode == '2':
        adapter_names = ["Base"]
        prog = tqdm(total=len(prompts) * (len(matrix_config['adapter_paths']) + 1))
        adapter_images = []
        # Base Model + character
        prog.set_description("Base")
        for prompt in prompts:
            prompt_ = (additional_triggers + ' ' + prompt).strip()
            output_jpeg_bytes = pipe.generate(prompt=prompt_, width=width, height=height,
                                              num_steps=matrix_config.get('sample_steps', 50), guidance=4, seed=None,
                                              silent=True)
            adapter_images.append(output_jpeg_bytes)
            prog.update()
        gen_images.append(adapter_images)


        for adap in range(len(matrix_config.get('adapter_paths', []))):
            adapter_images = []
            path = matrix_config['adapter_paths'][adap]
            if not os.path.exists(path):
                continue
            name = os.path.basename(path).split(('.safet'))[0]
            adap_trigger_path = os.path.join(os.path.dirname(path), name + '.txt')
            adap_trigger = ""
            if os.path.exists(adap_trigger_path):
                with open(adap_trigger_path, 'r') as f:
                    adap_trigger = f.read()
            adap_trigger = additional_triggers + " " + adap_trigger
            pipe.model.load_lora(path, 1.0, name, silent=True)
            adapter_names.append(name)
            prog.set_description(name)

            for prompt in prompts:
                prompt_ = adap_trigger + ' ' + prompt
                output_jpeg_bytes = pipe.generate(prompt=prompt_, width=width, height=height,
                                            num_steps=matrix_config.get('sample_steps', 50), guidance=4, seed=None, silent=True)
                adapter_images.append(output_jpeg_bytes)
                prog.update()
            pipe.unload_lora(name, silent=True)
            gen_images.append(adapter_images)

        ci = concat_images(gen_images, width, height)
        tci = add_titles_concat_images(adapter_names, ci, width, height)

    else:
        print('Wrong matrix mode', args.mode)
        exit()


    tci.save(os.path.join(config['process'][0]['training_folder'], config['name'], "sample_matrix.jpg"), "JPEG", quality=95)



