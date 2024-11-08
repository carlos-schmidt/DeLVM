import torch
import numpy as np
from transformers import AutoModel, GenerationConfig
from modeling_taming_vqgan import VQGANModel
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    InterpolationMode,
    ToTensor,
)

encode_transform = Compose(
    [
        Resize(256, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(256),
        ToTensor(),
    ]
)


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
        """Given a batch of images in the form of: batch_size, n_images = (b,n), returns the generated images by DeLVM.

        The images will get resized to (256,256) before the encoding step.

        Args:
            imgs (list(list(image)) (b,n)): batch of length b with n-1 context images each and 1 prompt
        """
        batch_size = len(batched_images)

        # ------------------------------------------------------------------------------------------------------#
        # -------------------------------------------- transform  ----------------------------------------------#
        # --------------------------------------------    and     ----------------------------------------------#
        # --------------------------------------------   encode   ----------------------------------------------#
        # ------------------------------------------------------------------------------------------------------#

        vqgan_ids = []
        for batch in batched_images:
            if type(batch) is not torch.Tensor:
                # n,c,h,w
                transformed_batch = []
                for img in batch:
                    # Resize (256, bilinear) -> CenterCrop -> ToTensor(0..256->0..1)
                    img = encode_transform(img).to(self.device)
                    # c,h,w
                    transformed_batch.append(img)
                # n,c,h,w
                batch_tensor = torch.stack(transformed_batch)
            else:
                # We already transformed!
                batch_tensor = batch

            _, indices = self.vqgan.encode(batch_tensor)

            vqgan_ids.append(indices)

        # seq_prompt is batch of n transformed images on device.

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
        # output_images = self._convert_decode_to_pil(output_images)

        # now, output_images are of type PIL.Image.Image
        return output_images


if __name__ == "__main__":

    lvm_path = "./models/llama_300m_hf"  # path to converted hf model
    vqgan_path = "./models/vqgan-f16-8192-laion"  # path to vqgan model
    model = DeLVM(vqgan_path=vqgan_path, llama_path=lvm_path, device="cpu")

    from dataset import InteractiveDataset
    from torchvision.transforms.functional import to_pil_image
    from torch.utils.data import DataLoader
    from os import makedirs

    resize = Compose(
        [
            Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
        ]
    )

    dataset = InteractiveDataset(
        "/home/carlos/VOC2012",
        "InteractionsMerged",
        "SegmentationSingleObjects",
        transform=resize,
    )
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)

    for i, batch in enumerate(dataloader):
        model_input = []
        for image, gt in zip(*batch.values()):
            model_input.append(image)
            model_input.append(gt)

        # remove last gt
        outputs = model.inference([torch.stack(model_input[:-1])])

        model_input.append(outputs[0])
        catted = torch.cat(model_input, dim=2)
        makedirs("./interactive_demo/", exist_ok=True)
        to_pil_image(catted).save(f"./interactive_demo/{i}.png")
