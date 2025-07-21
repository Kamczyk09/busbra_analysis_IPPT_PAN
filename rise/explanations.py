import torch.nn as nn
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import torch


class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=10):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_masks(self, N, s, p1, savepath='rise/masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size), dtype=np.float32)

        for i in tqdm(range(N), desc='Generating filters'):
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            self.masks[i] = resize(
                grid[i], up_size, order=1, mode='reflect', anti_aliasing=False
            )[x:x + self.input_size[0], y:y + self.input_size[1]]

        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)

        self.masks = torch.from_numpy(self.masks).float().to(self.device)
        self.N = N
        self.p1 = p1


    def load_masks(self, filepath):
        raw = np.load(filepath)

        if raw.ndim == 3:
            raw = raw[:, np.newaxis, :, :]  # Add channel dimension if missing

        self.masks = torch.from_numpy(raw).float().to(self.device)
        self.N = self.masks.shape[0]
        self.p1 = self.masks.mean().item()

        # print(f"[RISE] Loaded {self.N} masks of shape {self.masks.shape[2:]}")
        # print(f"[RISE] Mean activation (p1): {self.p1:.4f}")
        # print(f"[RISE] Tensor dtype: {self.masks.dtype}")

    # def forward(self, x):
    #     N = self.N
    #     x = x.to(self.device)
    #     _, _, H, W = x.size()
    #     stack = self.masks * x
    #
    #     preds = []
    #     for i in range(0, N, self.gpu_batch):
    #         preds.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
    #     p = torch.cat(preds)
    #
    #     CL = p.size(1)
    #     sal = torch.matmul(p.transpose(0, 1), self.masks.view(N, H * W))
    #     sal = sal.view(CL, H, W)
    #     sal = sal / N / self.p1
    #     return sal

    def forward(self, x):
        N = self.N
        x = x.to(self.device)
        _, _, H, W = x.size()
        stack = self.masks * x

        preds = []
        for i in range(0, N, self.gpu_batch):
            out = self.model(stack[i:min(i + self.gpu_batch, N)])

            # 🛠 Zapewnij, że wyjście ma kształt [batch, num_classes]
            if out.ndim == 1:
                out = out.unsqueeze(1)  # [N] → [N, 1]
            elif out.ndim == 2 and out.size(1) == 1:
                out = torch.sigmoid(out)  # Jeśli logit, przekształć na probabilistyczne
            elif out.ndim == 2:
                out = torch.softmax(out, dim=1)  # Dla wieloklasowych

            preds.append(out)

        p = torch.cat(preds)  # [N, CL]
        CL = p.size(1)

        sal = torch.matmul(p.transpose(0, 1), self.masks.view(N, H * W))  # [CL, H*W]
        sal = sal.view(CL, H, W)
        sal = sal / N / self.p1
        return sal


class RISEBatch(RISE):
    def forward(self, x):
        x = x.to(self.device)
        B, C, H, W = x.size()
        N = self.N
        stack = self.masks.view(N, 1, H, W) * x.view(1, B * C, H, W)
        stack = stack.view(B * N, C, H, W)

        preds = []
        for i in range(0, N * B, self.gpu_batch):
            preds.append(self.model(stack[i:min(i + self.gpu_batch, N * B)]))
        p = torch.cat(preds)

        CL = p.size(1)
        p = p.view(N, B, CL)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal


# To process in batches
# def explain_all_batch(data_loader, explainer):
#     n_batch = len(data_loader)
#     b_size = data_loader.batch_size
#     total = n_batch * b_size
#     # Get all predicted labels first
#     target = np.empty(total, 'int64')
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Predicting labels')):
#         p, c = torch.max(nn.Softmax(1)(explainer.model(imgs.cuda())), dim=1)
#         target[i * b_size:(i + 1) * b_size] = c
#     image_size = imgs.shape[-2:]
#
#     # Get saliency maps for all images in val loader
#     explanations = np.empty((total, *image_size))
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Explaining images')):
#         saliency_maps = explainer(imgs.cuda())
#         explanations[i * b_size:(i + 1) * b_size] = saliency_maps[
#             range(b_size), target[i * b_size:(i + 1) * b_size]].data.cpu().numpy()
#     return explanations
