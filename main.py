import fire
import numpy as np
import torch
import yaml
from helper import (
    accuracy,
    generate_weights,
    load_precomputed_features,
    set_seed,
)
from clip import clip # openAI' CLIP
from torchvision.transforms import v2 as T
from torchvision import datasets
from torch.nn import functional as F
from PIL import Image

import open_clip # openCLIP' CLIP
from open_CLIP.utils import get_engine, OPENCLIP_MODEL_DIC


def main(
    dataset_name: str = "imagenet",
    num_workers: int = 4,
    seed: int = 42,
    device: str = "cuda",
):
    device = torch.device(device)
    print("Device:", device)
    print("num_workers:", num_workers)

    # load config file
    with open(file=f"cfgs/{dataset_name}.yaml") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    set_seed(seed)

    # load hyperparameters from config file
    model_size = hparams["model_size"]
    alpha = hparams["alpha"]
    n_samples = hparams["n_samples"]
    batch_size = hparams["batch_size"]
    data_path = hparams["data_path"]
    pre_trained_corpus = hparams["pre_trained_corpus"]

    # load model
    print(f"Loading {model_size}")

    ################ openAI CLIP  ################
    # Notice: Model directly loaded to device
    # model, processor = clip.load(model_size, device=device)
    # model.eval()
    # model.requires_grad_(False)
    
    ################ openCLIP  ################
    # model, _, processor = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', 
    #                                                             pretrained='laion400m_e32',
    #                                                             device = device)
    # model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    # openCLIP_tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

    ################ openCLIP  get_engine() ################
    model, processor, tokenizer = get_engine(arch=model_size, corpus=pre_trained_corpus)
    model.cuda() # movel the model to cuda device

    def random_crop(image: Image.Image, alpha: float = 0.1) -> Image.Image:
        """Randomly crops an image within a size range determined by alpha and the image dimensions.

        Args:
            image (Image): The input image to crop.
            alpha (float): The minimum scale factor for the crop as a proportion of the smallest dimension.

        Returns:
            PIL Image or Tensor: Cropped image
        """
        # Get the width and height of the original image
        w, h = image.size
        # Determine the size of the crop based on alpha and the smallest dimension
        n_px = np.random.uniform(low=alpha, high=0.9) * min(h, w)
        # Perform the crop
        cropped = T.RandomCrop(int(n_px))(image)

        return cropped

    def custom_loader(path: str) -> torch.Tensor:
        """Loads an image, applies a processing function, and returns augmented versions.

        Args:
            path (str): The path to the image file.
            n_samples (int): The number of augmented samples to generate.

        Returns:
            torch.Tensor: A tensor stack of the processed image and its augmented samples.
        """
        # Load the image using the default loader
        img = datasets.folder.default_loader(path)
        # Process the image and generate additional augmented samples
        augmented_imgs = [processor(img)]
        augmented_imgs.extend(processor(random_crop(img)) for _ in range(n_samples))
        # Return a stacked tensor of all processed images
        return torch.stack(augmented_imgs)

    # pre-compute image features from dataset
    (
        precomputed_features,
        target,
        image_features,
    ) = load_precomputed_features(
        model,
        dataset_name=dataset_name,
        model_size=model_size,
        alpha=alpha,
        n_samples=n_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        data_path=data_path,
        custom_loader=custom_loader,
        device=device,
    )

    max_size = precomputed_features.size(1)
    image_features = image_features.to(device)

    results = {}
    with torch.no_grad():
        methods = hparams["methods"]
        for method in methods:
            method = list(method.values())[0]
            method_name = method["name"]
            method_enabled = method["enabled"]

            text_scale = (
                torch.exp(torch.tensor(method["text_scale"])).to(device)
                if "text_scale" in method
                else None
            )
            image_scale = (
                torch.exp(torch.tensor(method["image_scale"])).to(device)
                if "image_scale" in method
                else None
            )

            if method_enabled:
                zeroshot_weights = generate_weights(
                    method_name,
                    model=model,
                    dataset_name=dataset_name,
                    tt_scale=text_scale,
                    device=device,
                    tokenizer=tokenizer,
                )
                # set zero-shot weights to the same dtype as image features
                zeroshot_weights = zeroshot_weights.to(image_features.dtype)
            else:
                continue

            # Baseline
            logits = image_features.squeeze(1) @ zeroshot_weights
            baseline_acc = accuracy(
                logits, target, image_features.size(0), dataset_name
            )
            if method_name != "ours":
                print(f"{method_name}: {baseline_acc:.2f}\n")
                results[method_name] = round(baseline_acc, 2)

            if method_name == "ours":
                acc_list = []
                patch_num = hparams["patch_n"]
                print(f"n_run: {hparams['n_run']}")
                for i in range(hparams["n_run"]):
                    random_indices = torch.randint(0, max_size, (patch_num,))
                    sampled_features = precomputed_features[:, random_indices, :]

                    patch_embeds = sampled_features[:, :, :-1]
                    patch_weights = sampled_features[:, :, -1]
                    del sampled_features

                    # Weighted average of image embeddings
                    w_i = (patch_weights * image_scale).softmax(-1).unsqueeze(-1)
                    patch_embeds = (patch_embeds * w_i).sum(dim=1)
                    patch_embeds = F.normalize(patch_embeds, dim=-1)

                    # Ours: [B, D] @ [C, D].T -> (B, C)
                    logits = patch_embeds @ zeroshot_weights
                    acc_list.append(
                        accuracy(logits, target, patch_embeds.size(0), dataset_name)
                    )

                mean = np.mean(acc_list)
                std = np.std(acc_list)
                print(f"{method_name}: {mean:.2f}+-{std:.2f}")
                print(acc_list)


if __name__ == "__main__":
    fire.Fire(main)
