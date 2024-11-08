# Interactive Dataset Types:
# 1. (input, interaction, output)
# 2. (input+interaction, output)
from os import walk, path

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image

class InteractiveDataset(Dataset):

    def __init__(self, root_dir, image_dir, gt_dir, transform=None):
        self.image_dir = path.join(root_dir, image_dir)
        self.gt_dir = path.join(root_dir, gt_dir)

        self.filenames = next(walk(self.gt_dir), (None, None, []))[2]
        self.filenames = [fn.split(".")[0] for fn in self.filenames]
        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda x: to_tensor(x)

    def __len__(self):
        # We take GT image length since they might contain fewer images than input
        return len(self.filenames)

    def get_image(self, dir, image_name):
        files_in_dir = next(walk(dir), (None, None, []))[2]
        for file in files_in_dir:
            if image_name == file.split("/")[-1].split(".")[0]:
                return Image.open(path.join(dir, file))

        print(f"{image_name} not found in {dir}")
        return None

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = self.get_image(self.image_dir, filename)
        ground_truth = self.get_image(self.gt_dir, filename)

        image_transformed = self.transform(image)
        gt_transformed = self.transform(ground_truth)

        return {"image": image_transformed, "gt": gt_transformed}
