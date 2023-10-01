"""Extract CLIP visual feature of character profile images"""

import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from glob import glob 
import clip
import PIL
from PIL import Image
from glob import glob


class ImageFolder(Dataset):
    def __init__(self, data_root: str, preprocess):
        print(f'building dataset from {data_root} ...')
        self.data_root = data_root
        self.all_paths = sorted(glob(os.path.join(self.data_root, '*')))
        self.preprocess = preprocess
        self.dummy = torch.zeros(3, 224, 224)

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, index: int):
        fname = self.all_paths[index]
        image_path = f"{fname}"
        assert os.path.exists(image_path)
        is_error = False
        image = self.dummy
        try:
            image = self.preprocess(Image.open(image_path))
        except PIL.UnidentifiedImageError:
            is_error = True
        except OSError:
            is_error = True
        except BaseException:
            is_error = True
        if is_error:
            return image, "ERROR", os.path.basename(image_path)
        return image, 'YES', os.path.basename(image_path)


if __name__ == '__main__':
    # clip_model_type = 'ViT-B/32'
    clip_model_type = 'ViT-L/14'

    # model
    device = torch.device("cuda:0")
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    clip_model = clip_model.eval()

    # dataset
    image_root = '/scratch/shared/beegfs/htd/audiovault/actor_profiles'
    # available at: wget http://www.robots.ox.ac.uk/~htd/autoad/actor_profiles.tar
    ds = ImageFolder(image_root, preprocess)
    dl = torch.utils.data.DataLoader(ds, batch_size=200, shuffle=False, num_workers=8, drop_last=False)

    # main loop
    all_embeddings = []
    all_captions = []

    progress = tqdm(total=len(dl))
    counter = 0
    clip_model_name = clip_model_type.replace('/', '-')
    out_data_path = f"audiovault_face_{clip_model_name}.pth.tar"
    all_valid_mask = []

    for i, data in enumerate(dl):
        images, captions, image_names = data
        images = images.to(device)
        with torch.no_grad():
            feature = clip_model.encode_image(images).cpu()
        is_valid = list(map(lambda x: x != "ERROR", captions))
        mask = torch.tensor(is_valid)
        all_embeddings.append(feature[mask])
        image_names = [image_name for j, image_name in enumerate(image_names) if is_valid[j]]
        all_captions.extend(image_names)
        all_valid_mask.append(mask)
        progress.update()
        counter += len(image_names)

    all_valid_mask = torch.cat(all_valid_mask, dim=0)
    torch.save({"clip_embedding": torch.cat(all_embeddings, dim=0), "filenames": all_captions}, out_data_path)
    progress.close()
    print(f'finished extracting {clip_model_type} features from {image_root}')
    print(f'Success rate: {all_valid_mask.float().mean()}')
    assert torch.cat(all_embeddings, dim=0).shape[0] == len(all_captions)

    """
    python extract_clip_face.py
    """