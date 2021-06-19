import argparse
import cv2
import os
import glob
from tensorflow.python.ops.gen_array_ops import expand_dims
from model import WCT2
from utils import read_img, download_weight


VALID_EXTS = ['png', 'jpg', 'jpeg']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str)
    parser.add_argument('--style', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--checkpoint', type=str, default='pretrained')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=1.0)

    return parser.parse_args()

def check_path(args):
    assert os.path.exists(args.content), f'{args.content} does not exist'
    assert os.path.exists(args.style), f'{args.style} does not exist'

    is_c_dir = not os.path.isfile(args.content)
    is_s_dir = not os.path.isfile(args.style)

    assert (
        not (is_c_dir and is_s_dir),
        f'Only one of content or style can be a directory, the other should be a file')

    if not is_c_dir and not is_s_dir:
        out_ext = args.output.split('.')[-1]
        assert out_ext in VALID_EXTS, f'output must end with {VALID_EXTS}, got {out_ext}'
    else:
        # One is dir
        assert not os.path.isfile(args.output), f'Output must be a directory'
        os.makedirs(args.output, exist_ok=True)


def main(args):
    check_path(args)

    model = WCT2()
    weight = download_weight() if args.checkpoint == 'pretrained' else args.checkpoint
    model.load_weight(weight)

    content_imgs = [args.content]
    style_imgs = [args.style]

    if os.path.isdir(args.content):
        content_imgs = glob.glob(args.content)
    elif os.path.isdir(args.style):
        style_imgs = glob.glob(args.style)

    print(f'{len(content_imgs)} content images, {len(style_imgs)} style images')

    for cont in content_imgs:
        cont_img = read_img(cont, args.image_size, expand_dims=True)
        for sty in style_imgs:
            sty_img = read_img(sty, args.image_size, expand_dims=True)
            gen =  model.transfer(cont_img, sty_img, args.alpha)
            if cv2.imwrite(args.output, gen[0][...,::-1]):
                print(f'Saved image to {args.output}')


if __name__ == '__main__':
    main(parse_args())
