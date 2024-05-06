import os
import numpy as np
import scipy.misc as m
from PIL import Image
#import h5py
from torch.utils import data
from torchvision import transforms
from skimage.transform import resize
from dataloaders import custom_transforms as tr
import SimpleITK as sitk

class AtriaSegmentation(data.Dataset):
    NUM_CLASSES = 3


    def __init__(self, args, split="train"):

        self.root = args.inputDir
        self.split = split
        self.args = args
        self.files = {}

        #self.images_base = os.path.join(self.root, self.split)
        self.images_base = os.path.join(self.root)
        self.annotations_base = os.path.join(self.images_base, 'masks/')
        self.files[split] = '' #self.glob(rootdir=self.images_base, suffix='.h5')
        if not os.path.exists(args.inputFile):
            raise Exception("File %s not found." % (args.inputFile))
        #niiGlob = glob.glob(args.inputFile)
        inputNiiFile = args.inputFile
        inputImg = sitk.ReadImage(inputNiiFile)
        # Re-orient to LPS?
        scan_vol = sitk.GetArrayFromImage(inputImg)
        self.scanVolume = scan_vol

        self.valid_classes = [0, 1]
        self.class_names = ['unlabelled', 'HEART']

        #self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        #if not self.files[split]:
        #    raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        #print("Found %d %s images" % (len(self.files[split]), split))

        print("Found %d images in scan volume" % (self.scanVolume.shape[0]))


    def __len__(self):
        return self.scanVolume.shape[0] #len(self.files[self.split])

    def __getitem__(self, index):

        _img, _maskname, _imagesize = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'maskname': _maskname, 'imagesize': _imagesize}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)
        else: 
            return self.transform_ts(sample())

    def _make_img_gt_point_pair(self, index):

        #img_path = self.files[self.split][index].rstrip()        
        #dataset_dir, _fname = os.path.split(img_path)
        #mask_fname = _fname.replace("scan_1_", "")
        #lbl_path = os.path.join(self.annotations_base,
        #                        _fname)
        #_img, _imagesize = self._load_image(img_path)
        mask_fname = os.path.join(self.root, 'mask_dummy.nii')
        lbl_path = ''
        _img, _imagesize = self._load_image(index)
        _target = self._load_mask(lbl_path)

        return _img, mask_fname, _imagesize

    def _load_image(self, index):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        
        #hf = h5py.File(img_path, 'r')
        #im = hf['/scan'][:]
        #image = np.array(im)
        #imagesize = image.shape
        #imagesize = np.asarray(imagesize)
        #image = image.reshape(im.shape).transpose()


        # Read H5 image
        # hf = h5py.File(img_path, 'r')
        # datasetName = list(hf.keys())
        # if "scan" in datasetName:
        #     im = hf['/scan'][:]
        # else:
        #     if "OctaveExportScan" in datasetName:
        #         im = hf['/OctaveExportScan']
        #         im = im['value']
        #         im = im[:]
        #     else:
        #         raise Exception("Scan dataset not found.")
        #
        # image = np.array(im)
        # image = image.reshape(im.shape).transpose()
        image = self.scanVolume[index, :, :]
        imagesize = np.shape(image)

        # add intensity offset
        image = image + 1024

        #resize all images to 512x512
        image = resize(image, (512,512), anti_aliasing = True)

        #normalize image from 0-255 (as original pre-trained images were RGB between 0-255)
        image = (255*(image - np.min(image))/np.ptp(image).astype(int)).astype(np.uint8)

        #concating the image for all three channels
        if image.ndim != 3:
            image = np.dstack([image,image,image])

        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        image = Image.fromarray(image.astype(np.uint8))

        return image, imagesize

    def _load_mask(self, lbl_path):
        """Generate instance masks for an image.
       Returns:
        mask: A uint8 array of shape [height, width] with multiple labels in one mask file.
        """
        
        #hf = h5py.File(lbl_path, 'r')
        #m1 = hf['/mask'][:]
        m = np.zeros(shape=(512,512))
        #m = np.array(m1)
        #m = m.reshape(m1.shape).transpose()
        #deleting heart mask so that it is not trained
        m[m==1]=0
        m = Image.fromarray(m.astype(np.uint8))
        m = m.resize((512, 512), Image.NEAREST)
        # get label infoscanOffset

        # Return mask containing all labels existing on the image
        return m

    def glob(self, rootdir='.', suffix=''):
        """Performs glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(rootdir, filename)
            for filename in os.listdir(rootdir) if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            #tr.RandomHorizontalFlip(),
            tr.RandomRotate(degree=(90)),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.Normalize(mean=(0.316, 0.316, 0.316), std=(0.188, 0.188, 0.188)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.316, 0.316, 0.316), std=(0.188, 0.188, 0.188)),
            #maybe try min_max normalization between -1 and 1?
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.316, 0.316, 0.316), std=(0.188, 0.188, 0.188)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512
    args.crop_size = 512

    heart_train = HeartSegmentation(args, split='train')

    dataloader = DataLoader(heart_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='heart')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.188, 0.188, 0.188)
            img_tmp += (0.316, 0.316, 0.316)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)

