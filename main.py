import argparse

import yaml
from tensorboardX import SummaryWriter

from datasets import *
from solver import Solver


def main(args):
    with open("./config.yml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    train_logger = SummaryWriter(log_dir=os.path.join(config['log'], 'train'), comment='training')

    solver = Solver(config)

    train_trans = image_transform(**config['train_transform'])
    config['train_dataset'].update({'transform': train_trans,
                                    'target_transform': None,
                                    'categories': config['class_names']})

    if config.get('val_dataset') is not None:
        config['val_dataset'].update({'transform': train_trans,
                                      'target_transform': None,
                                      'categories': config['class_names']})

    if args.train_prm:
        config['train_dataset'].update({'train_type': 'prm'})
        dataset = train_dataset(**config['train_dataset'])
        config['data_loaders']['dataset'] = dataset
        data_loader = get_dataloader(**config['data_loaders'])

        if config.get('val_dataset') is not None:
            config['val_dataset'].update({'train_type': 'prm'})
            dataset = train_dataset(**config['val_dataset'])
            config['data_loaders']['dataset'] = dataset
            val_data_loader = get_dataloader(**config['data_loaders'])
        else:
            val_data_loader = None

        solver.train_prm(data_loader, train_logger, val_data_loader)
        print('train prm over')

    if args.train_filling:
        proposals_trans = proposals_transform(**config['train_transform'])
        config['train_dataset'].update({
            'train_type': 'filling',
            'target_transform': proposals_trans,
        })

        dataset = train_dataset(**config['train_dataset'])
        config['data_loaders']['dataset'] = dataset
        data_loader = get_dataloader(**config['data_loaders'])

        solver.train_filling(data_loader, train_logger)
        print('train filling over')

    if args.run_demo:
        test_trans = image_transform(**config['test_transform'])
        config['test_dataset'].update({'image_size': config['test_transform']['image_size'],
                                       'transform': test_trans})
        dataset = test_dataset(**config['test_dataset'])
        config['test_data_loaders']['dataset'] = dataset
        data_loader = get_dataloader(**config['test_data_loaders'])

        solver.inference(data_loader)
        print('predict over')


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_filling', type=bool, default=False, help='set train filling mode up')
    parser.add_argument('--train_prm', type=bool, default=False, help='set train prm mode up')
    parser.add_argument('--run_demo', '-I', type=bool, default=False, help='run demo')
    args = parser.parse_args()
    main(args)
