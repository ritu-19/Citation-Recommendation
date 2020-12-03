import numpy as np
import random as rn
from train import *
from preprocess import *
from args_parser import parse_args
from human_analysis import *



def main(args):
    print(args)
    rn.seed(321)
    np.random.seed(321)
    torch.manual_seed(321)
    torch.cuda.manual_seed(321)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using: ', device)

    preprocess = args.preprocess
    task = args.task
    dataset_path = args.data_path
    data_type = args.data_type
    if preprocess:
        preprocessCls = Preprocessing(dataset_path, task, data_type)
        preprocessCls.preprocess()
    if args.mode == "train":
        if task == "Classification":
            encodings_train = pickle.load(open("data/" + data_type + "/BERT" + task + "Encodings_train.pkl", 'rb'))
            labels_train = pickle.load(open("data/" + data_type + "/BERT" + task + "Labels_train.pkl", 'rb')).long()
            trainBERTClassification(encodings_train.to(device), labels_train.to(device), epochs=args.epoch,
                                    batch_size=args.batch_size, lr=args.learning_rate, lr_decay=args.lr_decay,
                                    step_size=args.step_size)

        else:
            encodings1 = pickle.load(open("data/" + data_type + "/BERT" + task + "Encodings1_train.pkl", 'rb'))
            labels = pickle.load(open("data/" + data_type + "/BERT" + task + "Labels_train.pkl", 'rb'))
            encodings2 = pickle.load(open("data/" + data_type + "/BERT" + task + "Encodings2_train.pkl", 'rb'))
            trainBERTContrastive(encodings1.to(device), encodings2.to(device), labels.to(device), epochs=args.epoch,
                                 batch_size=args.batch_size, lr=args.learning_rate, loss_type=args.loss, lr_decay=args.lr_decay,
                                    step_size=args.step_size)

    elif args.mode == "human_eval":

        if task == "Classification":
            path = args.pretrained_model
            model = BERTClassification().to(device)
            model.load_state_dict(torch.load(path))
            _ = human_eval_classification(model, mode='test', batch_size=args.batch_size, rows=1000)

        else:
            path = args.pretrained_model
            model = BERTContrastive().to(device)
            model.load_state_dict(torch.load(path))
            _ = human_eval_ranking(model, mode='test', batch_size=args.batch_size, rows=100)

    else:
        if task == "Classification":
            path = args.pretrained_model
            model = BERTClassification().to(device)
            model.load_state_dict(torch.load(path))
            _ = eval_contrastive(model, mode='test', batch_size=args.batch_size)

        else:
            path = args.pretrained_model
            model = BERTContrastive().to(device)
            model.load_state_dict(torch.load(path))
            _ = eval_contrastive(model, mode='test', batch_size=args.batch_size)


if __name__ == "__main__":
    args = parse_args()
    main(args)
