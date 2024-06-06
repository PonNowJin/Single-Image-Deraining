import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_new import RainRemovalDataset
import torch.nn.functional as F
from PIL import ImageEnhance
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import resize, to_pil_image, to_tensor

model_name = 'final_model_continue_residual.pth'
learning_rate = 1e-4
features = 32
epochs = 0
batches = 4
contrast_factor = 1.4
sharpness_factor = 1.8


class GuidedFilter(nn.Module):
    def __init__(self, radius, eps):
        super(GuidedFilter, self).__init__()
        self.radius = radius
        self.eps = eps

    def box_filter(self, x, r):
        """Box filter implementation."""
        ch = x.shape[1]
        kernel_size = 2 * r + 1
        box_kernel = torch.ones((ch, 1, kernel_size, kernel_size), dtype=x.dtype, device=x.device)
        return F.conv2d(x, box_kernel, padding=r, groups=ch)

    def forward(self, I, p):
        r = self.radius
        eps = self.eps
        N = self.box_filter(torch.ones_like(I), r)

        mean_I = self.box_filter(I, r) / N
        mean_p = self.box_filter(p, r) / N
        mean_Ip = self.box_filter(I * p, r) / N

        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.box_filter(I * I, r) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = self.box_filter(a, r) / N
        mean_b = self.box_filter(b, r) / N

        q = mean_a * I + mean_b
        return q


class RainRemovalNet(nn.Module):
    def __init__(self, num_features=features, num_channels=3, kernel_size=3):
        super(RainRemovalNet, self).__init__()
        self.guided_filter = GuidedFilter(15, 1)
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU()

        layers = []
        num_residual_blocks = 16
        for _ in range(num_residual_blocks):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size, padding=1))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU())

        self.residual_layers = nn.Sequential(*layers)
        self.conv_final = nn.Conv2d(num_features, num_channels, kernel_size, padding=1)
        self.bn_final = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        base = self.guided_filter(x, x)
        detail = x - base

        out = self.relu(self.bn1(self.conv1(detail)))
        out_shortcut = out

        for i in range(16):
            out = self.residual_layers[3 * i](out)
            out = self.residual_layers[3 * i + 1](out)
            out = self.residual_layers[3 * i + 2](out)
            out_shortcut = out_shortcut + out

        neg_residual = self.bn_final(self.conv_final(out_shortcut))
        final_out = x + neg_residual

        return final_out, base, detail


device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = RainRemovalDataset('rainy_image_dataset/training', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=True)
val_dataset = RainRemovalDataset('rainy_image_dataset/testing', train=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batches, shuffle=False)

net = RainRemovalNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

model_path = model_name
if os.path.exists(model_path):
    print("Load model successfully.")
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print("Train new model.")

num_epochs = epochs
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    for i, (inputs, labels, original_size) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    torch.save(net.state_dict(), f'rain_removal_epoch_{epoch+1}.pth')

torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_name)

print('Finished Training')

net.eval()
val_loss = 0.0
with torch.no_grad():
    for i, (inputs, labels, original_size) in enumerate(val_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, base, detail = net(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        sizes = []
        for x in range(batches):
            sizes.append((int(original_size[1][x]), int(original_size[0][x])))

        # print(sizes)

        for j in range(outputs.size(0)):
            output_image = outputs[j].cpu()
            base_image = base[j].cpu()
            detail_image = detail[j].cpu()

            output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
            base_image = (base_image - base_image.min()) / (base_image.max() - base_image.min())
            detail_image = (detail_image - detail_image.min()) / (detail_image.max() - detail_image.min())

            output_image = resize(output_image, sizes[j])
            base_image = resize(base_image, sizes[j])
            detail_image = resize(detail_image, sizes[j])

            # 轉為PIL圖像
            output_image_pil = to_pil_image(output_image)
            base_image_pil = to_pil_image(base_image)
            detail_image_pil = to_pil_image(detail_image)

            # 增強對比度
            enhancer = ImageEnhance.Contrast(output_image_pil)
            output_image_pil = enhancer.enhance(contrast_factor)

            # 增強銳利度
            enhancer_sharpness = ImageEnhance.Sharpness(output_image_pil)
            output_image_pil = enhancer_sharpness.enhance(sharpness_factor)

            # 轉回 Tensor
            output_image = to_tensor(output_image_pil)
            base_image = to_tensor(base_image_pil)
            detail_image = to_tensor(detail_image_pil)

            output_dir = 'Train_IEEE_output_train_3'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'image_{i * val_loader.batch_size + j + 1}.png')
            base_path = os.path.join(output_dir, f'image_{i * val_loader.batch_size + j + 1}_base.png')
            detail_path = os.path.join(output_dir, f'image_{i * val_loader.batch_size + j + 1}_detail.png')
            vutils.save_image(output_image, output_path)
            vutils.save_image(base_image, base_path)
            vutils.save_image(detail_image, detail_path)
            print(f'Saved: {output_path}')
            print(f'Saved base image: {base_path}')
            print(f'Saved detail image: {detail_path}')

val_loss /= len(val_loader)
print(f'Validation Loss: {val_loss:.4f}')
