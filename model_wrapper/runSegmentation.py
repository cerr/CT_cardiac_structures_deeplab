import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import h5py
from dataloaders import make_data_loader
from modeling.deeplab import *
from PIL import Image
from skimage.transform import resize
import time
import glob
import SimpleITK as sitk
from dataloaders import custom_transforms as tr
from torchvision import transforms


class Trainer(object):
    def __init__(self, argv):    # Define paths

        num_args = len(argv)
        # if num_args == 2: # container
        #     batchSize = int(sys.argv[1])
        #     inputNiiPath = '/scratch/inputH5/'
        #     outputNiiPath = '/scratch/outputH5/'
        #     resnetDir = '/software/model/resnet'
        #     modelWeights = '/software/model/heart_atria_model.pth'
        # else:
        inputNiiPath = argv[1]
        outputNiiPath = argv[2]
        #batchSize = int(argv[3])
        modelAndWeight = argv[3]
        scriptDir = os.path.dirname(os.path.abspath(__file__))
        wrapperDir = os.path.join(scriptDir, os.pardir)
        #modelDir = os.path.join(wrapperDir, os.pardir, 'model')
        modelDir = os.path.join(wrapperDir, 'model')
        resnetDir = os.path.join(modelDir, 'resnet')
        modelWeights = os.path.join(modelDir, modelAndWeight[1]) # 'heart_atria_model.pth'

        batchSize = 1
        self.args = argv
        self.crop_size = 513
        self.cuda = False

        self.dataset = modelAndWeight[0] # 'atria'
        self.validate_model = modelWeights #'/software/model/heart_atria_model.pth'
        self.validate_out_folder = outputNiiPath #'/scratch/outputH5/'  # relative to container
        self.inputDir = os.path.dirname(inputNiiPath) #'/scratch/inputH5/'
        self.inputFile = inputNiiPath

        self.batch_size = batchSize #int(sys.argv[1])
        self.workers = 0 #1

        os.environ['TORCH_HOME'] = resnetDir #'/software/model/resnet'
        #os.environ['KMP_DUPLICATE_LIB_OK']='True'

        # Define Dataloader
        kwargs = {'num_workers': self.workers, 'pin_memory': True}
        self.test_loader, self.nclass = make_data_loader(self, **kwargs)
        num_img_val = len(self.test_loader)

        # Define network
        self.model = DeepLab(num_classes=self.nclass,
                        backbone='resnet',
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)

        # loading model for validation
        if self.validate_model is not None:
            if not os.path.isfile(self.validate_model):
                raise RuntimeError("=> no checkpoint found at '{}'".format(self.validate_model))

        print("GPU DEVICE COUNT")
        print(torch.cuda.device_count())
        
        print("Cuda Available?")
        print(torch.cuda.is_available())

        # Using cuda
        if torch.cuda.device_count() and torch.cuda.is_available():
            self.cuda = True
            print('Using GPU...')
            tDataP = time.time()

            device = torch.device("cuda:0")
            self.model = self.model.to(device)
            checkpoint = torch.load(self.validate_model)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            elapsed = time.time() - tDataP
            print("***Model Load Time***")
            print(elapsed)

        else:
            self.cuda = False
            print('Using CPU...')
            checkpoint = torch.load(self.validate_model, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint['state_dict'])

    def validation(self, model):
        self.model.eval()

        ### Initialize array to hold model output in same shape as input original_scan_vol
        input_size = self.test_loader.dataset.scanVolume.shape
        labels_out = np.zeros((input_size[0], input_size[1], input_size[2]))

        ### Begin processing
        print('Starting inference...')

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tbar = tqdm(self.test_loader, desc='\r')
        
        for i, sample in enumerate(tbar):
            image = sample['image']
            fname = sample['maskname']
            imageSize = sample['imagesize']

            if self.cuda:
                image = image.cuda()

            with torch.no_grad():
                output = self.model(image)

            test = torch.max(output, 1)[1].detach().cpu().numpy()
            out_image = Image.fromarray(test[0,].astype(np.int16))
            out_image = out_image.resize(imageSize[::-1],
                                         Image.Resampling.NEAREST)  # PIL resize uses (W,H) whereas numpy shape is (H,W). hence swap W and H in imageSize

            labels_out[i] = out_image #out_image.resize(imageSize[::-1], Image.Resampling.NEAREST) # PIL resize uses (W,H) whereas numpy shape is (H,W). hence swap W and H in imageSize

        ### Save output
        #maskOut = np.moveaxis(labels_out, [1, 2], [2, 1])
        #maskOut = np.flip(np.flip(maskOut, axis=1), axis=2)
        maskOut = labels_out

        inputFileName = os.path.basename(self.inputFile)
        outFilePrefix, ext = os.path.splitext(os.path.splitext(inputFileName)[0])
        #outFilePrefix = outFilePrefix.replace('scan', 'MASK')
        outFile = os.path.join(self.validate_out_folder, outFilePrefix + '_' + model[0] + '.nii.gz')
        try:
            maskImg = sitk.GetImageFromArray(maskOut)
            inputImg = sitk.ReadImage(self.inputFile)
            maskImg.CopyInformation(inputImg)
            sitk.WriteImage(maskImg, outFile)
        except:
            Exception('Unable to save Nii file output.')


def main(argv):
    t = time.time()
    niiGlob = glob.glob(os.path.join(argv[1], '*.nii.gz'))
    inputNiiFile = niiGlob[0]
    argv[1] = inputNiiFile
    argv.append('model name')
    models = [('atria', 'heart_atria_model.pth'),
              ('heart', 'heart_checkpoint.pth'),
              ('heartStructure', 'heart_struct_model_best.pth'),
              ('pericardium', 'heart_peri_model.pth'),
              ('ventricles', 'heart_ventricles_model.pth')]
    print('Starting Inference:')
    for model in models:
        argv[-1] = model
        trainer = Trainer(argv)
        trainer.validation(model)
    elapsed = time.time() - t
    print("***Total Time***")
    print(elapsed)

if __name__ == "__main__":
    main(sys.argv)
