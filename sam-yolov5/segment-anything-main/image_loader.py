def open(fp, mode='r'):
    """
    Opens and identifies the given image file.
    """
    if mode != 'r':
        raise ValueError("Mode must be 'r'")
    if hasattr(fp, 'read'):
        fp = fp.read()
    image = ImgFile(fp)
    try:
        image.load()
    except OSError as e:
        raise UnidentifiedImageError("Failed to load image; %s" % e) from e

    return image

class ImageDataset(Dataset):
    """Dataset for loading images and targets.

    Attributes:
        img_folder (str): Path to the folder containing the images.
        target_folder (str): Path to the folder containing the xyxys.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g., ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, img_folder, target_folder, transform=None, target_transform=None):
        self.img_folder = img_folder  # 图片文件路径
        self.target_folder = target_folder  # xyyx文件路径
        self.transform = transform
        self.target_transform = target_transform

        # Get image filenames
        self.img_filenames = os.listdir(img_folder)

        # Get target filenames
        self.target_filenames = os.listdir(target_folder)

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_folder, img_filename)
        img = Image.open(img_path)

        # Load target
        target_filename = self.target_filenames[idx]
        target_path = os.path.join(self.target_folder, target_filename)
        target = np.load(target_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
