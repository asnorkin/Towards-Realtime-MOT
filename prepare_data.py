import os
import os.path as osp
from argparse import ArgumentParser
from shutil import copy
from sklearn.model_selection import train_test_split


def config_data_sources(args):
    prefixes = [
        'inter_panoramic_5splits_',
        'inter_autofollow_duo_',
    ]

    return [
        {
            'images_dir': osp.join(args.data_dir, 'images/train'),
            'labels_dir': osp.join(args.data_dir, 'player_labels_with_ids/train'),
            'prefix': prefix,
        }
        for prefix in prefixes
    ]


def config():
    ap = ArgumentParser()

    ap.add_argument('--data_dir', default='../footballobjectdetection/data')
    ap.add_argument('--experiment_dir', default='experiments/train')
    ap.add_argument('--val_size', type=float, default=0.1)

    args = ap.parse_args()
    return args


def save_images_list(files, dst):
    with open(dst, 'w+') as outf:
        outf.write('\n'.join(files))


def main(args):
    if osp.exists(args.experiment_dir):
        os.removedirs(args.experiment_dir)

    datadir = osp.join(args.experiment_dir, 'data')
    os.makedirs(datadir)

    data_sources = config_data_sources(args)

    result = []
    for source in data_sources:
        image_fnames = [fname for fname in os.listdir(source['images_dir'])
                        if fname.startswith(source['prefix'])]
        print('Found {} images with prefix {}'.format(len(image_fnames), source['prefix']))
        for image_fname in image_fnames:
            name, ext = osp.splitext(image_fname)
            assert ext == '.jpg'
            image_file_src = osp.join(source['images_dir'], name + '.jpg')
            image_file = osp.join(datadir, name + '.jpg')
            copy(image_file_src, image_file)

            result.append(image_file)

            label_file_src = osp.join(source['labels_dir'], name + '.txt')
            label_file = osp.join(datadir, name + '.txt')
            copy(label_file_src, label_file)

            labels = []
            with open(label_file) as inpf:
                for line in inpf:
                    clid, objid, *coords = line.strip().split()
                    if clid in {'0', '3'}:
                        labels.append(' '.join(['0', objid] + coords))

            with open(label_file, 'w+') as outf:
                outf.write('\n'.join(labels))

    indices = list(range(len(result)))
    train_indices, val_indices = train_test_split(indices, test_size=args.val_size)
    train, val = [result[i] for i in train_indices], [result[i] for i in val_indices]

    save_images_list(train, osp.join(args.experiment_dir, 'inter.train'))
    save_images_list(val, osp.join(args.experiment_dir, 'inter.val'))


if __name__ == '__main__':
    args = config()
    main(args)
