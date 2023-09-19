# reference to https://github.com/naver-ai/DenseDiffusion
import gradio as gr
import numpy as np
import os
import pickle
import base64
import argparse
import ast
import time
from io import BytesIO
from PIL import Image
from typing import List

from omegaconf import OmegaConf

import mindspore as ms
from mindspore import ops, Tensor

from gm.models.diffusion import DiffusionEngine
from gm.modules.diffusionmodules.sampler import EulerEDMSampler
from gm.modules.diffusionmodules.sampling_utils import to_d
from gm.helpers import \
    get_batch, get_unique_embedder_keys_from_conditioner, \
    SD_XL_BASE_RATIOS, VERSION2SPECS, create_model, perform_save_locally, embed_watermark
from gm.util import append_dims, seed_everything

MAX_COLORS = 12


#################################################
#################################################
canvas_html = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
load_js = """
async () => {
const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/sketch-canvas.js"
fetch(url)
  .then(res => res.text())
  .then(text => {
    const script = document.createElement('script');
    script.type = "module"
    script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
    document.head.appendChild(script);
  });
}
"""

get_js_colors = """
async (canvasData) => {
  const canvasEl = document.getElementById("canvas-root");
  return [canvasEl._data]
}
"""

css = '''
#color-bg{display:flex;justify-content: center;align-items: center;}
.color-bg-item{width: 100%; height: 32px}
#main_button{width:100%}
<style>
'''


def create_binary_matrix(img_arr, target_color):
    mask = np.all(img_arr == target_color, axis=-1)
    binary_matrix = mask.astype(int)
    return binary_matrix


def preprocess_mask(mask_, h, w):
    mask = np.array(mask_)
    mask = mask.astype(np.float32)
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = Tensor(mask, ms.float16)
    mask = ops.ResizeNearestNeighbor(size=(h, w))(mask)
    return mask


def process_sketch(canvas_data):
    binary_matrixes = []
    base64_img = canvas_data['image']
    image_data = base64.b64decode(base64_img.split(',')[1])
    image = Image.open(BytesIO(image_data)).convert("RGB")
    im2arr = np.array(image)
    colors = [tuple(map(int, rgb[4:-1].split(','))) for rgb in canvas_data['colors']]
    colors_fixed = []

    r, g, b = 255, 255, 255
    binary_matrix = create_binary_matrix(im2arr, (r, g, b))  # get background binary_matrix
    binary_matrixes.append(binary_matrix)
    binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
    colored_map = binary_matrix_ * (r, g, b) + (1 - binary_matrix_) * (50, 50, 50)
    colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))

    for color in colors:
        r, g, b = color
        if any(c != 255 for c in (r, g, b)):
            binary_matrix = create_binary_matrix(im2arr, (r, g, b))
            binary_matrixes.append(binary_matrix)
            binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
            colored_map = binary_matrix_ * (r, g, b) + (1 - binary_matrix_) * (50, 50, 50)
            colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))

    visibilities = []
    colors = []
    for n in range(MAX_COLORS):
        visibilities.append(gr.update(visible=False))
        colors.append(gr.update())
    for n in range(len(colors_fixed)):
        visibilities[n] = gr.update(visible=True)
        colors[n] = colors_fixed[n]

    return [gr.update(visible=True), binary_matrixes, *visibilities, *colors]


def process_prompts(binary_matrixes, *seg_prompts):
    return [gr.update(visible=True), gr.update(value=' , '.join(seg_prompts[:len(binary_matrixes)]))]


def process_example(layout_path, all_prompts, seed_):
    all_prompts = all_prompts.split('***')

    binary_matrixes = []
    colors_fixed = []

    im2arr = np.array(Image.open(layout_path))[:, :, :3]
    unique, counts = np.unique(np.reshape(im2arr, (-1, 3)), axis=0, return_counts=True)
    sorted_idx = np.argsort(-counts)

    binary_matrix = create_binary_matrix(im2arr, (0, 0, 0))
    binary_matrixes.append(binary_matrix)
    binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
    colored_map = binary_matrix_ * (255, 255, 255) + (1 - binary_matrix_) * (50, 50, 50)
    colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))

    for i in range(len(all_prompts) - 1):
        r, g, b = unique[sorted_idx[i]]
        if any(c != 255 for c in (r, g, b)) and any(c != 0 for c in (r, g, b)):
            binary_matrix = create_binary_matrix(im2arr, (r, g, b))
            binary_matrixes.append(binary_matrix)
            binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
            colored_map = binary_matrix_ * (r, g, b) + (1 - binary_matrix_) * (50, 50, 50)
            colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))

    visibilities = []
    colors = []
    prompts = []
    for n in range(MAX_COLORS):
        visibilities.append(gr.update(visible=False))
        colors.append(gr.update())
        prompts.append(gr.update())

    for n in range(len(colors_fixed)):
        visibilities[n] = gr.update(visible=True)
        colors[n] = colors_fixed[n]
        prompts[n] = all_prompts[n + 1]

    return [gr.update(visible=True), binary_matrixes, *visibilities, *colors, *prompts,
            gr.update(visible=True), gr.update(value=all_prompts[0]), int(seed_)]


class DiffusionEngineDenseDiff(DiffusionEngine):
    def preprocess_condition(
            self,
            value_dict, num_samples, force_uc_zero_embeddings, dtype,
            binary_matrixes, creg_, sreg_, sizereg_, latent_size, master_prompt, prompts
    ):
        if num_samples[0] != 1:
            raise ValueError("num_samples not ")

        # bsz = 1
        creg, sreg, sizereg = creg_, sreg_, sizereg_
        sp_sz, sp_sz = latent_size[:2]

        clipped_prompts = prompts[:len(binary_matrixes)]
        prompts = [master_prompt] + list(clipped_prompts)
        layouts = ops.concat([preprocess_mask(mask_, sp_sz, sp_sz) for mask_ in binary_matrixes])

        # get condition
        value_dict["prompt"] = master_prompt
        batch, batch_uc = get_batch(
            get_unique_embedder_keys_from_conditioner(self.conditioner),
            value_dict,
            num_samples,
            dtype=dtype
        )
        for key in batch:
            if isinstance(batch[key], Tensor):
                print(key, batch[key].shape)
            elif isinstance(batch[key], list):
                print(key, [len(i) for i in batch[key]])
            else:
                print(key, batch[key])
        print("Get Condition Done.")

        # get additional condition
        additional_batches = []
        for p in prompts[1:]:
            value_dict["prompt"] = p
            a_batch, _ = get_batch(
                get_unique_embedder_keys_from_conditioner(self.conditioner),
                value_dict,
                [1,],
                dtype=dtype
            )
            additional_batches.append(a_batch)
        print("Get Additional Condition Done.")

        print("Embedding Starting...")
        conditioner = self.conditioner
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = []
        for embedder in conditioner.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0

        # embedding
        uncond_embeddings = conditioner(batch if batch_uc is None else batch_uc, force_uc_zero_embeddings)
        # cond_embeddings = conditioner(batch)
        tokens, lengths = conditioner.tokenize(batch)
        tokens_tensor = [Tensor(t) for t in tokens]
        cond_embeddings = conditioner.embedding(tokens_tensor)

        # additional embedding
        additional_cond_embeddings, additional_tokens, additional_lengths = [], [], []
        for _batch in additional_batches:
            _tokens, _lengths = conditioner.tokenize(_batch)
            _tokens_tensor = [Tensor(t) for t in tokens]
            _emb = conditioner.embedding(_tokens_tensor)
            additional_cond_embeddings.append(_emb)
            additional_tokens.append(_tokens)
            additional_lengths.append(_lengths)

        for embedder, rate in zip(conditioner.embedders, ucg_rates):
            embedder.ucg_rate = rate
        print("Embedding Done.")

        ###########################
        ###### prep for sreg ######
        ###########################
        sreg_maps = {}
        reg_sizes = {}
        for r in range(4):
            res = int(sp_sz / np.power(2, r))
            layouts_s = ops.ResizeNearestNeighbor(size=(res, res))(layouts)
            layouts_s = (
                    layouts_s.view(layouts_s.shape[0], 1, -1) *
                    layouts_s.view(layouts_s.shape[0], -1, 1)
            ).sum(0).unsqueeze(0)  #.repeat((bsz, 1, 1))

            res_pow_2 = int(np.power(res, 2))
            reg_sizes[res_pow_2] = 1 - sizereg * layouts_s.sum(-1, keepdims=True) / res_pow_2
            sreg_maps[res_pow_2] = layouts_s

        ###########################
        ###### prep for creg ######
        ###########################
        pww_maps = ops.zeros((1, 77, sp_sz, sp_sz), ms.float16)
        for i in range(len(additional_tokens)):
            additional_token, additional_length = additional_tokens[i], additional_lengths[i]
            additional_cond_embedding = additional_cond_embeddings[i]

            token = tokens[0]
            assert lengths[0] is not None
            assert token.shape[0] == 1
            assert len(cond_embeddings['crossattn'].shape) == 3 and cond_embeddings['crossattn'].shape[0] == 1
            assert additional_length[0] == additional_length[1], \
                f"Expect embedder1 and embedder2 to have the same token length"
            wlen = int(additional_length[0] - 2)
            widx = additional_token[0][0, 1: 1 + wlen]

            _replace_token_flag = False
            for j in range(77):
                if np.array(token[0, j:j + wlen] == widx).sum() == wlen:
                    pww_maps[:, j:j + wlen, :, :] = layouts[i:i + 1]
                    cond_embeddings['crossattn'][0, j: j + wlen] = \
                        additional_cond_embedding['crossattn'][0, 1: 1 + wlen]
                    _replace_token_flag = True
                    break
            if not _replace_token_flag:
                raise gr.Error("Please check whether every segment prompt is included in the full text !")

        creg_maps = {}
        for r in range(4):
            res = int(sp_sz / np.power(2, r))
            layout_c = ops.ResizeNearestNeighbor(size=(res, res))(pww_maps)
            layout_c = layout_c.view(1, 77, -1).transpose(0, 2, 1)  #.repeat((bsz, 1, 1))

            res_pow_2 = int(np.power(res, 2))
            creg_maps[res_pow_2] = layout_c

        return cond_embeddings, uncond_embeddings, sreg_maps, creg_maps, reg_sizes, creg, sreg

    def do_sample_dense_diffusion(
        self,
        sampler,
        value_dict,
        num_samples,
        H,
        W,
        C,
        F,
        force_uc_zero_embeddings: List = None,
        batch2model_input: List = None,
        return_latents=False,
        filter=None,
        amp_level="O0",
        **kwargs
    ):
        dtype = ms.float32 if amp_level not in ("O2", "O3") else ms.float16
        seed = kwargs.pop('seed_', None)
        if seed:
            np.random.seed(seed)

        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []

        num_samples = [num_samples]
        c, uc, sreg_maps, creg_maps, reg_sizes, creg, sreg = self.preprocess_condition(
            value_dict, num_samples, force_uc_zero_embeddings, dtype, **kwargs
        )

        for k in c:
            if not k == "crossattn":
                c[k], uc[k] = map(
                    lambda y: y[k][: int(np.prod(num_samples))],
                    (c, uc)
                )

        shape = (np.prod(num_samples), C, H // F, W // F)
        randn = Tensor(np.random.randn(*shape), ms.float32)

        print("Sample latent Starting...")
        samples_z = sampler(
            self, randn, cond=c, uc=uc,
            sreg=sreg, creg=creg, creg_maps=creg_maps, sreg_maps=sreg_maps, reg_sizes=reg_sizes
        )
        print("Sample latent Done.")

        print("Decode latent Starting...")
        samples_x = self.decode_first_stage(samples_z)
        samples_x = samples_x.asnumpy()
        print("Decode latent Done.")

        samples = np.clip((samples_x + 1.0) / 2.0, a_min=0.0, a_max=1.0)

        if filter is not None:
            print("Filter Starting...")
            samples = filter(samples)
            print("Filter Done.")

        if return_latents:
            return samples, samples_z
        return samples


class EulerEDMSamplerDenseDiff(EulerEDMSampler):

    def denoise(self, x, model, sigma, cond, uc, **kwargs):
        noised_input, sigmas, cond = self.guider.prepare_inputs(x, sigma, cond, uc)
        cond = model.openai_input_warpper(cond)
        c_skip, c_out, c_in, c_noise = model.denoiser(sigmas, noised_input.ndim)
        model_output = model.model(
            ops.cast(noised_input * c_in, ms.float32),
            ops.cast(c_noise, ms.int32),
            **cond,
            **kwargs
        )
        model_output = model_output.astype(ms.float32)
        denoised = model_output * c_out + noised_input * c_skip
        denoised = self.guider(denoised, sigma)
        return denoised

    def sampler_step(self, sigma, next_sigma, model, x, cond, uc=None, gamma=0.0, **kwargs):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = Tensor(np.random.randn(*x.shape), x.dtype) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, model, sigma_hat, cond, uc, **kwargs)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, model, cond, uc)
        return x

    def __call__(self, model, x, cond, uc=None, num_steps=None, **kwargs):
        x = ops.cast(x, ms.float32)
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                model,
                x,
                cond,
                uc,
                gamma,
                sample_steps=i,
                **kwargs
            )

        return x


def init_sampling(
    steps=40,
    sampler="EulerEDMSamplerDenseDiff",
    guider="VanillaCFG",
    discretization="LegacyDDPMDiscretization",
    img2img_strength=1.0,
    stage2strength=None,
):
    from gm.modules.diffusionmodules.discretizer import Img2ImgDiscretizationWrapper, Txt2NoisyDiscretizationWrapper
    from gm.helpers import get_guider, get_discretization
    assert sampler == "EulerEDMSamplerDenseDiff"
    assert guider in ["VanillaCFG", "IdentityGuider"]
    assert discretization in [
        "LegacyDDPMDiscretization",
        "EDMDiscretization",
    ]

    steps = min(max(steps, 1), 1000)
    guider_config = get_guider(guider)
    discretization_config = get_discretization(discretization)

    s_churn, s_tmin, s_tmax, s_noise = 0.0, 0.0, 999.0, 1.0
    sampler = EulerEDMSamplerDenseDiff(
        num_steps=steps,
        discretization_config=discretization_config,
        guider_config=guider_config,
        s_churn=s_churn,
        s_tmin=s_tmin,
        s_tmax=s_tmax,
        s_noise=s_noise,
        verbose=True,
    )

    if img2img_strength < 1.0:
        print(f"WARNING: Wrapping {sampler.__class__.__name__} with Img2ImgDiscretizationWrapper")
        sampler.discretization = Img2ImgDiscretizationWrapper(sampler.discretization, strength=img2img_strength)
    if stage2strength is not None:
        sampler.discretization = Txt2NoisyDiscretizationWrapper(
            sampler.discretization, strength=stage2strength, original_steps=steps
        )

    return sampler


def run_txt2img(
    args, model, version_dict, is_legacy=False, return_latents=False, filter=None, stage2strength=None, amp_level="O0"
):
    assert args.sd_xl_base_ratios in SD_XL_BASE_RATIOS
    W, H = SD_XL_BASE_RATIOS[args.sd_xl_base_ratios]
    C = version_dict["C"]
    F = version_dict["f"]
    force_uc_zero_embeddings = ["txt"] if not is_legacy else []
    batch2model_input = None

    with open(args.example_prompts, 'rb') as f:
        val_prompt = pickle.load(f)
    val_layout = args.example_layouts

    num_samples = 1
    value_dict = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "orig_width": args.orig_width if args.orig_width else W,
        "orig_height": args.orig_height if args.orig_height else H,
        "target_width": args.target_width if args.target_width else W,
        "target_height": args.target_height if args.target_height else H,
        "crop_coords_top": max(args.crop_coords_top if args.crop_coords_top else 0, 0),
        "crop_coords_left": max(args.crop_coords_left if args.crop_coords_left else 0, 0),
        "aesthetic_score": args.aesthetic_score if args.aesthetic_score else 6.0,
        "negative_aesthetic_score": args.negative_aesthetic_score if args.negative_aesthetic_score else 2.5,
    }
    sampler = init_sampling(
        sampler=args.sampler,
        guider=args.guider,
        discretization=args.discretization,
        steps=args.sample_step,
        stage2strength=stage2strength,
    )

    # model, sampler, value_dict, num_samples, H, W, C, F, force_uc_zero_embeddings, batch2model_input, return_latents, filter, amp_level,
    def process_generation(binary_matrixes, seed_, creg_, sreg_, sizereg_, master_prompt, *prompts):
        print("Txt2Img Sampling")
        s_time = time.time()

        samples = model.do_sample_dense_diffusion(
            sampler,
            value_dict,
            num_samples,
            H, W, C, F,
            force_uc_zero_embeddings,
            batch2model_input,
            return_latents,
            filter,
            amp_level,

            binary_matrixes=binary_matrixes, seed_=seed_, creg_=creg_, sreg_=sreg_, sizereg_=sizereg_,
            latent_size=(H // F, W // F), master_prompt=master_prompt, prompts=prompts
        )
        samples = samples[0] if isinstance(samples, (list, tuple)) else samples

        images = []
        samples = embed_watermark(samples)
        for sample in samples:
            sample = 255.0 * sample.transpose(1, 2, 0)
            images.append(Image.fromarray(sample.astype(np.uint8)))

        print(f"Txt2Img sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")

        return (images)

    #################################################
    #################################################
    ### gradio context
    with gr.Blocks(css=css) as demo:
        binary_matrixes = gr.State([])
        color_layout = gr.State([])
        gr.Markdown('''## DenseDiffusion (SDXL)''')
        gr.Markdown('''
        #### Reference to https://github.com/naver-ai/DenseDiffusion <br>
        #### 😺 Instruction to generate images 😺 <br>
        (1) Create the image layout. <br>
        (2) Label each segment with a text prompt. <br>
        (3) Adjust the full text. The default full text is automatically concatenated from each segment's text. The default one works well, but refineing the full text will further improve the result. <br>
        (4) Check the generated images, and tune the hyperparameters if needed. <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - w<sup>c</sup> : The degree of attention modulation at cross-attention layers. <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - w<sup>s</sup> : The degree of attention modulation at self-attention layers. <br>
        ''')

        with gr.Row():
            with gr.Box(elem_id="main-image"):
                canvas_data = gr.JSON(value={}, visible=False)
                canvas = gr.HTML(canvas_html)
                button_run = gr.Button("(1) I've finished my sketch ! 😺", elem_id="main_button", interactive=True)

                prompts = []
                colors = []
                color_row = [None] * MAX_COLORS
                with gr.Column(visible=False) as post_sketch:
                    for n in range(MAX_COLORS):
                        if n == 0:
                            with gr.Row(visible=False) as color_row[n]:
                                colors.append(
                                    gr.Image(shape=(100, 100), label="background", type="pil", image_mode="RGB",
                                             width=100, height=100)
                                )
                                prompts.append(gr.Textbox(label="Prompt for the background (white region)", value=""))
                        else:
                            with gr.Row(visible=False) as color_row[n]:
                                colors.append(
                                    gr.Image(shape=(100, 100), label="segment " + str(n), type="pil", image_mode="RGB",
                                             width=100, height=100))
                                prompts.append(gr.Textbox(label="Prompt for the segment " + str(n)))

                    get_genprompt_run = gr.Button("(2) I've finished segment labeling ! 😺", elem_id="prompt_button",
                                                  interactive=True)

                with gr.Column(visible=False) as gen_prompt_vis:
                    general_prompt = gr.Textbox(value='', label="(3) Textual Description for the entire image",
                                                interactive=True)
                    with gr.Accordion("(4) Tune the hyperparameters", open=False):
                        creg_ = gr.Slider(
                            label=" w\u1D9C (The degree of attention modulation at cross-attention layers) ", minimum=0,
                            maximum=2., value=1.5, step=0.1)
                        sreg_ = gr.Slider(
                            label=" w \u02E2 (The degree of attention modulation at self-attention layers) ", minimum=0,
                            maximum=2., value=1.0, step=0.1)
                        sizereg_ = gr.Slider(label="The degree of mask-area adaptive adjustment", minimum=0, maximum=1.,
                                             value=1., step=0.1)
                        # bsz_ = gr.Slider(label="Number of Samples to generate", minimum=1, maximum=4, value=1, step=1)
                        seed_ = gr.Slider(label="Seed", minimum=-1, maximum=999999999, value=42, step=1)

                    final_run_btn = gr.Button("Generate ! 😺")

                    layout_path = gr.Textbox(label="layout_path", visible=False)
                    all_prompts = gr.Textbox(label="all_prompts", visible=False)

            with gr.Column():
                out_image = gr.Gallery(label="Result", columns=2, height='auto')

        button_run.click(process_sketch, inputs=[canvas_data],
                         outputs=[post_sketch, binary_matrixes, *color_row, *colors], _js=get_js_colors, queue=False)

        get_genprompt_run.click(process_prompts, inputs=[binary_matrixes, *prompts],
                                outputs=[gen_prompt_vis, general_prompt], queue=False)

        final_run_btn.click(process_generation,
                            inputs=[
                                binary_matrixes, seed_, creg_, sreg_, sizereg_, general_prompt, *prompts,
                            ],
                            outputs=out_image)

        gr.Examples(
            examples=[[val_layout + '0.png',
                       '***'.join([val_prompt[0]['textual_condition']] + val_prompt[0]['segment_descriptions']), 42],
                      [val_layout + '1.png',
                       '***'.join([val_prompt[1]['textual_condition']] + val_prompt[1]['segment_descriptions']), 42],
                      [val_layout + '5.png',
                       '***'.join([val_prompt[5]['textual_condition']] + val_prompt[5]['segment_descriptions']), 42]],
            inputs=[layout_path, all_prompts, seed_],
            outputs=[post_sketch, binary_matrixes, *color_row, *colors, *prompts, gen_prompt_vis, general_prompt,
                     seed_],
            fn=process_example,
            run_on_click=True,
            label='😺 Examples 😺',
        )

        demo.load(None, None, None, _js=load_js)

    demo.launch(server_name="0.0.0.0")


def get_parser_sample():
    parser = argparse.ArgumentParser(description="sampling with sd-xl dense diffusion")

    parser.add_argument("--task", type=str, default="txt2img", choices=["txt2img", "img2img"])
    parser.add_argument("--config", type=str, default="configs/inference/sd_xl_dense_diffusion.yaml")
    parser.add_argument("--weight", type=str, default="checkpoints/sd_xl_base_1.0_ms.ckpt")
    parser.add_argument("--example_prompts", type=str, default="./demo/dataset/valset.pkl")
    parser.add_argument("--example_layouts", type=str, default="./demo/dataset/valset_layout/")

    parser.add_argument(
        "--prompt", type=str, default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--sd_xl_base_ratios", type=str, default="1.0")
    parser.add_argument("--orig_width", type=int, default=None)
    parser.add_argument("--orig_height", type=int, default=None)
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--crop_coords_top", type=int, default=None)
    parser.add_argument("--crop_coords_left", type=int, default=None)
    parser.add_argument("--aesthetic_score", type=float, default=None)
    parser.add_argument("--negative_aesthetic_score", type=float, default=None)
    parser.add_argument("--sampler", type=str, default="EulerEDMSamplerDenseDiff")
    parser.add_argument("--guider", type=str, default="VanillaCFG")
    parser.add_argument("--discretization", type=str, default="LegacyDDPMDiscretization")
    parser.add_argument("--sample_step", type=int, default=40)
    parser.add_argument("--num_cols", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="outputs/demo/", help="save dir")

    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument(
        "--ms_mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)"
    )
    parser.add_argument("--ms_jit", type=ast.literal_eval, default=True, help="use jit or not")
    parser.add_argument("--ms_amp_level", type=str, default="O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )

    # args for ModelArts
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument(
        "--ckpt_url", type=str, default="", help="ModelArts: obs path to pretrain model checkpoint file"
    )
    parser.add_argument("--train_url", type=str, default="", help="ModelArts: obs path to output folder")
    parser.add_argument(
        "--multi_data_url", type=str, default="", help="ModelArts: list of obs paths to multi-dataset folders"
    )
    parser.add_argument(
        "--pretrain_url", type=str, default="", help="ModelArts: list of obs paths to multi-pretrain model files"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/cache/pretrain_ckpt/",
        help="ModelArts: local device path to checkpoint folder",
    )
    return parser


def sample(args):
    config = OmegaConf.load(args.config)
    version = config.pop("version", "SDXL-base-1.0")
    version_dict = VERSION2SPECS.get(version)
    task = args.task
    add_pipeline = False

    seed_everything(args.seed)

    # Init Model
    model, filter = create_model(
        config, checkpoints=args.weight.split(","), freeze=True, load_filter=False, amp_level=args.ms_amp_level
    )  # TODO: Add filter support

    save_path = os.path.join(args.save_path, task, version)
    is_legacy = version_dict["is_legacy"]
    args.negative_prompt = args.negative_prompt if is_legacy else ""

    stage2strength = None

    if task == "txt2img":
        out = run_txt2img(
            args,
            model,
            version_dict,
            is_legacy=is_legacy,
            return_latents=add_pipeline,
            filter=filter,
            stage2strength=stage2strength,
            amp_level=args.ms_amp_level,
        )
    elif task == "img2img":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown task {task}")

    out = out if isinstance(out, (tuple, list)) else [out, None]
    (samples, samples_z) = out

    perform_save_locally(save_path, samples)


def main():
    parser = get_parser_sample()
    args, _ = parser.parse_known_args()
    ms.context.set_context(mode=args.ms_mode, device_target=args.device_target)

    if args.ms_mode == 1:
        ms.context.set_context(pynative_synchronize=True)
    sample(args)


if __name__ == '__main__':
    main()
