import argparse
import cv2
from model import WCT2
from utils import read_img, download_weight


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str)
    parser.add_argument('--style', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--checkpoint', type=str, default='pretrained')
    parser.add_argument('--image-size', type=int, default=512)

    return parser.parse_args()


def main(args):
    model = WCT2()
    weight = download_weight() if args.checkpoint == 'pretrained' else args.checkpoint
    model.load_weight(weight)

    content = read_img(args.content, args.image_size)
    style = read_img(args.style, args.image_size)

    gen = model.transfer(content, style, 0.8)
    cv2.imwrite(args.output, gen[0] / 255.0)


if __name__ == '__main__':
    main(parse_args())
