import os
from PIL import Image
from torch.utils.data import Dataset


class RainRemovalDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.rain_imgs_path = os.path.join(self.root_dir, 'rainy_image')
        self.rain_imgs = [f for f in os.listdir(self.rain_imgs_path) if f.endswith('.jpg') and not f.startswith('.')]
        self.clean_imgs_path = os.path.join(self.root_dir, 'ground_truth')
        self.num_rain_images_per_clean_image = 14

    def __len__(self):
        return len(self.rain_imgs)

    def __getitem__(self, idx):
        rain_img_name = os.path.join(self.rain_imgs_path, self.rain_imgs[idx])

        clean_img_idx = int(self.rain_imgs[idx].split('_')[0])
        clean_img_name = os.path.join(self.clean_imgs_path, f"{clean_img_idx}.jpg")

        rain_image = Image.open(rain_img_name).convert('RGB')
        clean_image = Image.open(clean_img_name).convert('RGB')

        original_size = rain_image.size  # (width, height)

        if self.transform:
            rain_image = self.transform(rain_image)
            clean_image = self.transform(clean_image)

        if self.train:
            return rain_image, clean_image, original_size
        else:
            return rain_image, original_size
