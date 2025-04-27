# interface.py

from model import LeNet as TheModel
from train import my_descriptively_named_train_function as the_trainer
from predict import cryptic_inf_f as the_predictor
from dataset import FER2013Dataset as TheDataset
from dataset import unicornLoader as the_dataloader
from config import batch_size as the_batch_size
from config import epochs as total_epochs
