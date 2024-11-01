import torch
import numpy as np
from transformers import AutoModel, GenerationConfig
from InternLM.tools.model_hf.muse import VQGANModel
from InternLM.tools.utils import encode_transform


class DeLVM:

    def __init__(self, vqgan_path, llama_path, device="cuda"):
        self.vqgan: VQGANModel = VQGANModel.from_pretrained(vqgan_path)
        self.vqgan.to(device).eval()

        self.llama = AutoModel.from_pretrained(llama_path, trust_remote_code=True)
        self.llama.to(device).eval()

        self.device = device

        self.generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            num_beams=1,
            early_stopping=True,
            max_new_tokens=256,
        )

    def _convert_decode_to_pil(self, reconstructed_image):
        # Some magic i copied from the repo. Basically clamp to 0..1 then times 255.
        reconstructed_image = 2.0 * reconstructed_image - 1.0
        reconstructed_image = torch.clamp(reconstructed_image, -1.0, 1.0)
        reconstructed_image = (reconstructed_image + 1.0) / 2.0
        reconstructed_image *= 255.0

        # TODO do we want this on cpu?
        # (b,h,w,c)
        reconstructed_image = (
            reconstructed_image.permute(0, 2, 3, 1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        return reconstructed_image

    def inference(self, batched_images):
        """Given a batch of images in the form of:
        [
            {ctx_in_1, ctx_gt_1, ctx_in_2, ctx_gt_2, ..., prompt_in}
        ]

        returns the generated images by DeLVM.

        Args:
            imgs (numpy array (b,n,c,h,w)): batch of length b with n-1 context images each and 1 prompt
        """
        batch_size = len(batched_images)

        # Resize (256, bilinear) -> CenterCrop -> ToTensor(0..256->0..1)
        seq_prompt = []
        for batch in batched_images:
            # n,c,h,w
            transformed_batch = []
            for img in batch:
                # c,h,w
                img = encode_transform(img)
                img = img[0:3, :, :].to(self.device)
                transformed_batch.append(img)
            transformed_batch = torch.stack(transformed_batch)
            seq_prompt.append(transformed_batch)

        # seq_prompt is batch of n transformed images on device.

        # ------------------------------------------------------------------------------------------------------#
        # --------------------------------------------   encode   ----------------------------------------------#
        # ------------------------------------------------------------------------------------------------------#
        vqgan_ids = []
        for batch in seq_prompt:
            # pixels -> indices
            _, indices = self.vqgan.encode(batch)
            vqgan_ids.append(indices)

        vqgan_ids = torch.stack(vqgan_ids)
        vqgan_ids = torch.flatten(vqgan_ids, 1, 2)
        # ------------------------------------------------------------------------------------------------------#
        # --------------------------------------------  generate  ----------------------------------------------#
        # ------------------------------------------------------------------------------------------------------#
        with torch.no_grad():
            outputs = self.llama.generate(
                input_ids=vqgan_ids,
                generation_config=self.generation_config,
                max_new_tokens=256,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # ------------------------------------------------------------------------------------------------------#
        # --------------------------------------------   decode   ----------------------------------------------#
        # ------------------------------------------------------------------------------------------------------#

        # For each output, only select the generated image (indices)
        generated_ids = outputs.sequences[:, -256:]

        # Get VQGAN IDS which the transformer predicted
        generated_tokens = self.vqgan.quantize.get_codebook_entry_for_lvm(generated_ids)

        # btw: codebook dim is 64. shape of generated_tokens is (b,64,16,16)

        # Unflatten generated tokens and bring into proper image form (b,c,h,w). n omitted since n=1
        generated_tokens = generated_tokens.view(
            batch_size, generated_tokens.shape[1] // 16, 16, -1
        ).permute(0, 3, 1, 2)

        # decode indices to tokens
        output_images = self.vqgan.decode(generated_tokens)
        # shape of output_pixels is now (b,3,h,w) with h,w same as input

        # ------------------------------------------------------------------------------------------------------#
        # --------------------------------------------  to image  ----------------------------------------------#
        # ------------------------------------------------------------------------------------------------------#

        # since "pixels" are still roughly in range[0,1], convert to 0..255
        output_images = self._convert_decode_to_pil(output_images)

        # now, output_images are of type PIL.Image.Image
        return output_images


import os
from PIL import Image
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")  # TODO careful with that

prompt_path_1 = "./InternLM/tools/data/seg_prompt_1"  # path to prompt 1
prompt_path_2 = "./InternLM/tools/data/seg_prompt_2"  # path to prompt 2
lvm_path = "./models/llama_300m_hf"  # path to converted hf model
vqgan_path = "./models/vqgan-f16-8192-laion"  # path to vqgan model

# prepare prompt 1
img_names_1 = os.listdir(prompt_path_1)
img_names_1 = sorted(img_names_1)

seq_prompt_1 = []
for i, img_name in enumerate(img_names_1):
    # print('prompt: ', img_name)
    img_path = os.path.join(prompt_path_1, img_name)

    image = Image.open(img_path)
    seq_prompt_1.append(image)

batch = [seq_prompt_1]

# prepare prompt 2
img_names_2 = os.listdir(prompt_path_2)
img_names_2 = sorted(img_names_2)

seq_prompt_2 = []
for i, img_name in enumerate(img_names_2):
    img_path = os.path.join(prompt_path_2, img_name)

    image = Image.open(img_path)
    seq_prompt_2.append(image)

batch.append(seq_prompt_2)

delvm = DeLVM(vqgan_path=vqgan_path, llama_path=lvm_path, device="cpu")

outputs = delvm.inference(batch)

# From here on is only validation that we have the correct output
outputs = [Image.fromarray(o) for o in outputs]

fig, axes = plt.subplots(len(outputs), 2, figsize=(8, 4))

[axes[k,0].imshow(batch[k][-1]) for k in range(len(batch))]
[axes[k,1].imshow(outputs[k]) for k in range(len(outputs))]

plt.show()