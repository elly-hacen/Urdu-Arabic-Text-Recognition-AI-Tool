import os
import pytz
import math
import argparse
from PIL import Image
from datetime import datetime
import torch
from model import Model
from dataset import NormalizePAD
from utils import CTCLabelConverter, Logger

def read(opt, device):
    opt.device = device
    os.makedirs("read_outputs", exist_ok=True)
    datetime_now = str(datetime.now(pytz.timezone('Asia/Karachi')).strftime("%Y-%m-%d_%H-%M-%S"))
    logger = Logger(f'read_outputs/{datetime_now}.txt')
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    logger.log('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = model.to(device)

    # load model
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    logger.log('Loaded pretrained model from %s' % opt.saved_model)
    model.eval()
    
    if opt.rgb:
        img = Image.open(opt.image_path).convert('RGB')
    else:
        img = Image.open(opt.image_path).convert('L')
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(opt.imgH * ratio) > opt.imgW:
        resized_w = opt.imgW
    else:
        resized_w = math.ceil(opt.imgH * ratio)
    img = img.resize((resized_w, opt.imgH), Image.Resampling.BICUBIC)
    transform = NormalizePAD((1, opt.imgH, opt.imgW))
    img = transform(img)
    img = img.unsqueeze(0)
    # print(img.shape) # torch.Size([1, 1, 32, 400])
    batch_size = img.shape[0] # 1
    img = img.to(device)
    preds = model(img)
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
    
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    
    logger.log(preds_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Image Inference
    parser.add_argument('--image_path', required=True, help='Path to input image for inference')
    parser.add_argument('--saved_model', required=True, help='Path to trained model to load')

    # Input Image Config
    parser.add_argument('--batch_max_length', type=int, default=100, help='Maximum text label length')
    parser.add_argument('--imgH', type=int, default=32, help='Image height')
    parser.add_argument('--imgW', type=int, default=400, help='Image width')
    parser.add_argument('--rgb', action='store_true', help='Use RGB input (default: grayscale)')

    # Model Architecture - Locked to HRNet + DBiLSTM + CTC
    parser.add_argument('--FeatureExtraction', type=str, default="HRNet", help='(Locked) Feature extractor')
    parser.add_argument('--SequenceModeling', type=str, default="DBiLSTM", help='(Locked) Sequence modeling')
    parser.add_argument('--Prediction', type=str, default="CTC", help='(Locked) Prediction method')
    parser.add_argument('--num_fiducial', type=int, default=20, help='Number of fiducial points (only relevant for TPS if used)')
    parser.add_argument('--input_channel', type=int, default=1, help='Input channels for the model (1 for grayscale)')
    parser.add_argument('--output_channel', type=int, default=512, help='Output channels from feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of LSTM layers')

    # Device Selection
    parser.add_argument('--device_id', type=str, default=None, help='CUDA device ID if GPU is available')

    opt = parser.parse_args()

    # Force output_channel override for HRNet
    if opt.FeatureExtraction == "HRNet":
        opt.output_channel = 32

    # Load character set
    with open("UrduGlyphs.txt", "r", encoding="utf-8") as file:
        opt.character = ''.join([line.strip() for line in file.readlines()]) + " "

    # Device Setup
    cuda_str = 'cuda'
    if opt.device_id is not None:
        cuda_str = f'cuda:{opt.device_id}'
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)

    read(opt, device)
