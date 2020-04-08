import cv2 as cv
import glob
import numpy as np
import os
import os.path as osp
from argparse import ArgumentParser
from shutil import rmtree


def config_data_sources(args):
    names = [
        'atletico_autofollow',
        'atm_lev_autofollow',
        'cag_ver_autofollow',
        'inter_panoramic',
        'inter_autofollow',
        'mci-tot_autofollow',
        'mci_autofollow',
        'real_barca_autofollow',
        'roma_autofollow',
        'roma_crotone_autofollow',
    ]

    return [
        {
            'images_dir': osp.join(args.data_dir, 'images', name),
            'labels_dir': osp.join(args.data_dir, 'player_labels_extended', name),
            'prefix': name + '_',
        }
        for name in names
    ]


def config():
    ap = ArgumentParser()

    ap.add_argument('--data_dir', default='../footballobjectdetection/data')
    ap.add_argument('--experiment_dir', default='experiments/train')
    ap.add_argument('--min_overlap', type=int, default=40)
    ap.add_argument('--n_splits', type=int, default=None)
    ap.add_argument('--val_size', type=float, default=0.1)

    args = ap.parse_args()
    return args


def save_images_list(files, dst):
    with open(dst, 'w+') as outf:
        outf.write('\n'.join(files))


def read(source):
    image_prefix = osp.join(source['images_dir'], source['prefix'])
    image_files = glob.glob(image_prefix + '*.jpg')
    print('Found {} images with prefix {}'.format(len(image_files), source['prefix']))

    image_indices = list(map(lambda x: int(x[len(image_prefix):].split('.')[0]), image_files))
    for index in image_indices:
        image_file = '{}{}.jpg'.format(image_prefix, index)
        image = cv.imread(image_file)

        label_file = osp.join(source['labels_dir'], source['prefix'] + str(index) + '.txt')
        if osp.exists(label_file):
            label = []
            with open(label_file, 'r') as inpf:
                for line in inpf:
                    line = line.strip().split()
                    label.append([int(line[0])] + list(map(float, line[1:])))

            yield image, label, index


def split(full_image, full_label, n_splits=None, min_overlap=40):
    H, W = full_image.shape[:2]

    def overlap(n):
        return int(np.floor((n * H - W) / (n - 1)))

    if n_splits is None:
        n_splits = 2
        while overlap(n_splits) < min_overlap:
            n_splits += 1

    split.overlap = overlap(n_splits)
    split.n_splits = n_splits

    shift = H - split.overlap
    for split_index in range(1, n_splits + 1):
        x, w = (split_index - 1) * shift, H
        image_split = full_image[:, x: x + w]
        if full_label is None:
            yield image_split, None, split_index, x
            continue

        x_from, x_to, x_w = x / W, (x + w) / W, w / W
        label_split = []
        for clid, xc, yc, w, h in full_label:
            if x_from <= xc <= x_to:
                label_split.append([clid, (xc - x_from) / x_w, yc, w / x_w, h])

        yield image_split, label_split, split_index, x


def main(args):
    if osp.exists(args.experiment_dir):
        rmtree(args.experiment_dir)

    datadir = osp.join(args.experiment_dir, 'data')
    os.makedirs(datadir)

    data_sources = config_data_sources(args)

    result = {'train': [], 'val': []}
    for source_index, source in enumerate(data_sources):
        for full_image, full_label, full_index in read(source):
            val = np.random.binomial(1, args.val_size, size=1).astype(np.bool)
            key = 'val' if val else 'train'
            for split_image, split_label, split_index, _split_x in split(full_image, full_label, args.n_splits, args.min_overlap):
                name = '{}{}_{}'.format(source['prefix'], full_index, split_index)
                image_file = osp.join(datadir, name + '.jpg')
                cv.imwrite(image_file, split_image)

                clids = [lbl[0] for lbl in split_label]
                old = len(clids) > 0 and max(clids) < 8

                label_file = osp.join(datadir, name + '.txt')
                with open(label_file, 'w+') as outf:
                    for clid, *coords in split_label:
                        if clid in {1, 2}:
                            continue

                        obj_id = -1 if old else source_index * 10 + clid
                        clid = 0

                        outline = [str(clid), str(obj_id)] + ['{:.6f}'.format(max(coord, 0.000001))
                                                              for coord in coords]
                        outf.write(' '.join(outline) + '\n')

                result[key].append(image_file)

    save_images_list(result['train'], osp.join(args.experiment_dir, 'inter.train'))
    save_images_list(result['val'], osp.join(args.experiment_dir, 'inter.val'))


if __name__ == '__main__':
    args = config()
    main(args)
