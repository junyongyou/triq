from vit_iqa.modeling_vit_iqa import VisionTransformer, CONFIGS
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms


# file_name = '21852937'

def load_model():
    # Prepare model
    config = CONFIGS['ViT-B_16']

    num_classes = 5

    model = VisionTransformer(config, zero_head=True, num_classes=num_classes, load_transformer_weights=True, vis=True)

    model_checkpoint = r'C:\vq_datasets\results\ViT_patch\test0_checkpoint.bin'
    model.load_state_dict(torch.load(model_checkpoint))
    # torch.load(model_checkpoint)

    model.to('cpu')
    model.eval()

    return model


def main(file_name):
# if __name__ == '__main__':
    model = load_model()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image_file = r'.\database\train\koniq_small\{}.jpg'.format(file_name)
    im = Image.open(image_file)
    x = transform(im)

    logits, att_mat = model(x.unsqueeze(0))

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    v_mask = v[0, 1:]
    # grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    # v_mask = v[0, 1:][: grid_size * grid_size]
    mask = v_mask.reshape(12, 16).detach().numpy()

    # mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

    # mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")

    result_image = Image.fromarray(result)
    result_image.save(r'.\database\attention_masks\{}_mask_triq.png'.format(file_name))
    # result_image.save(r'.\database\train\202351926_mask_0.png')

    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    #
    # ax1.set_title('Original')
    # ax2.set_title('Attention Map')
    # _ = ax1.imshow(im)
    # _ = ax2.imshow(result)

    t = 0
