
import torch
import numpy as np

from PIL import Image

from .model_architecture import ArtifactAwareUNet


class ReliefLinkRestoration:

    def __init__(self, model_path):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.model = ArtifactAwareUNet(
            num_artifacts=7
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint["model_state_dict"]

        elif (
            isinstance(checkpoint, dict)
            and "state_dict" in checkpoint
        ):
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

        self.model.to(self.device)
        self.model.eval()


    @torch.no_grad()
    def restore(self, image):

        if isinstance(image, Image.Image):
            image = np.asarray(image)

        image = np.asarray(
            image,
            dtype=np.float32
        )

        if image.max() > 1.0:
            image = image / 255.0

        if image.ndim == 2:

            image = np.stack(
                [image, image, image],
                axis=-1
            )

        if image.ndim != 3 or image.shape[-1] != 3:

            raise ValueError(
                "Input must be an RGB image."
            )

        tensor = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float()

        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        restored, artifact_logits = self.model(
            tensor
        )

        restored = restored.squeeze(0)
        restored = restored.cpu().numpy()
        restored = restored.transpose(1, 2, 0)

        restored = np.clip(
            restored,
            0,
            1
        )

        return (
            restored,
            artifact_logits.cpu().numpy()
        )
