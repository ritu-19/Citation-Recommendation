import argparse


def parse_args():
    parser = argparse.ArgumentParser('Citation Recommendation System')
    parser.add_argument('--task', type=str, default='Classification', help='BERT Task Type: Classification/Contrastive [default: classification]')
    parser.add_argument('--loss', type=str, default='contrastive',
                        help='Loss Type for BERTContrastive: contrastive/cosine_embedding/margin_ranking [default: contrastive]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size During Draining [default: 8]')
    parser.add_argument('--epoch',  default=251, type=int, help='Epochs To Run [default: 251]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial Learning Rate [default: 0.001]')
    parser.add_argument('--mode', default="train", type=str, help='Evaluation Mode: train/test [default: train]')
    parser.add_argument('--pretrained_model', default='BERT_model_lr-5.pth', type=str, help='Pretrained Model Path: train/test [default: train]')
    parser.add_argument('--data_path', default='data/data.csv', type=str, help='Dataset Path')
    parser.add_argument('--preprocess', action='store_true', default=False, help='Dataset Preprocessing [True/False]')
    parser.add_argument('--step_size', type=int, default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
    return parser.parse_args()
