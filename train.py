import argparse
from model import WCT2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-tfrec', type=str)
    parser.add_argument('--val-tfrec', type=str, default='')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--checkpoint-path', type=str, default='/content/checkpoints/wtc2.h')
    parser.add_argument('--save-image-dir', type=str, default='/content/images')
    parser.add_argument('--resume', type=str, default='False')
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--debug-samples', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)

    return parser.parse_args()


def main(args):
    model = WCT2(lr=args.lr, save_interval=args.save_interval,
        gram_loss_weight=1.0, checkpoint_path=args.checkpoint_path)

    model.wct.load_weights(model.checkpoint_path)
    model.train(args.train_tfrec, args.val_tfrec, epochs=args.epochs,
        batch_size=args.batch_size)


if __name__ == '__main__':
    main(parse_args())