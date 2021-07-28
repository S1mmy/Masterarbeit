'''
Trainiert verschiedene GANs.
Samplet alle 25 oder 50 Epochs ein Bild und speichert dieses in results/
Einzelne Modelle werden aus dem Ordner models geladen.
MNIST oder SVHN Datenset wird automatisch über Tensorflow geladen.
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow_datasets as tfds

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="The max training epoch.", type=int, default=1000)
parser.add_argument("--batch_size", help="The batch size for training.", type=int, default=512)
parser.add_argument("--model", help="The Training Model (gan, dcgan, wgan, dragan)", type=str, default="GAN")
parser.add_argument("--dataset", help="The Dataset to train on (mnist, fashion_mnist, celeb_a, svhn)", type=str, default="mnist")
ARGS = parser.parse_args()


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

if ARGS.model.upper() == "GAN":
	from models.gan import GAN
elif ARGS.model.upper() == "DCGAN":
	from models.dcgan import GAN
elif ARGS.model.upper() == "WGAN":
	from models.wgan import GAN
elif ARGS.model.upper() == "DRAGAN":
	from models.dragan import GAN


if __name__ == "__main__":
	
	ARGS.dataset = ARGS.dataset.lower()
	if ARGS.dataset == "svhn":
		ARGS.dataset = "svhn_cropped"
	
	data, info = tfds.load(ARGS.dataset, with_info=True, data_dir='data/tensorflow_datasets')
	
	result_path = 'results/'+ARGS.model.upper()+'/'+ARGS.dataset+'/'+str(ARGS.batch_size)

	if not os.path.exists(result_path):
		os.makedirs(result_path)
		
	data_size, channel = 0, 0
	
	if ARGS.dataset == "mnist" or ARGS.dataset == "fashion_mnist":
		data_size, channel = 28, 1
	elif ARGS.dataset == "svhn_cropped":
		data_size, channel = 32, 3
		

	GAN = GAN(data['train'], ARGS.epochs, ARGS.batch_size, result_path, data_size, channel)
	if ARGS.epochs > 0:
		GAN.train()