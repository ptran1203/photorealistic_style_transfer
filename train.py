import argparse
import os
from model import WCT2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-tfrec', type=str)
    parser.add_argument('--val-tfrec', type=str, default='')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--checkpoint-path', type=str, default='/content/checkpoints/wtc2.h5')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)

    return parser.parse_args()


def main(args):
    model = WCT2(lr=args.lr, gram_loss_weight=1.0, checkpoint_path=args.checkpoint_path)

    if args.resume:
        if os.path.isfile(args.checkpoint_path):
            model.load_weight()
        else:
            model.load_weight('pretrained')

    model.train(args.train_tfrec, args.val_tfrec, epochs=args.epochs,
        batch_size=args.batch_size)


if __name__ == '__main__':
    main(parse_args())