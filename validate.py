import argparse
import os
import scipy
from SR_datasets import DatasetFactory
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time

description = 'SI Super Resolution pytorch implementation'

parser = argparse.ArgumentParser(description=description)

parser.add_argument('-m', '--model', metavar='M', type=str, default='dnESPCN',
                    help='network architecture. Default False')
parser.add_argument('-s', '--scale', metavar='S', type=int, default=4,
                    help='interpolation scale. Default 3')
parser.add_argument('--test-set', metavar='NAME', type=str, default='valid_4m',
                    help='dataset for testing. Default valid_4m')
args = parser.parse_args()

def display_config():
    print('############################################################')
    print('# SI Super Resolution - Pytorch implementation             #')
    print('# by Juan Luis Gonzalez Bello                              #')
    print('############################################################')
    print('')
    print('-------YOUR SETTINGS_________')
    for arg in vars(args):
        print("%15s: %s" % (str(arg), str(getattr(args, arg))))
    print('')

def export(scale, model_name, outputs, imgname):
    path = os.path.join('results_val1', model_name, str(scale) + 'x')

    if not os.path.exists(path):
        os.makedirs(path)

    for i, img in enumerate(outputs):
        img_name = os.path.join(path, imgname)
        scipy.misc.imsave(img_name, img.transpose(1, 2, 0))
        break

def main():
    display_config()
    useGPU = True

    print('Contructing dataset...')
    dataset_root = os.path.join('preprocessed_data', args.test_set)
    dataset_factory = DatasetFactory()
    val_dataset = dataset_factory.create_dataset('VALID', dataset_root)

    print('Loading model...')
    model_path = os.path.join('check_point', args.model, str(args.scale) + 'x', 'model.pt')
    if not os.path.exists(model_path):
        raise Exception('Cannot find %s.' % model_path)
    model = torch.load(model_path)
    if useGPU:
        model = model.cuda()
    else:
        model = model.cpu()

    # needed for forward pass! save memory
    for param in model.parameters():
        param.requires_grad = False

    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=6)

    print('Testing...')
    avrtime = 0
    cnt = 0
    for i, (input_batch, imgname) in enumerate(dataloader):
        if useGPU:
            input_batch = Variable(input_batch.cuda(), requires_grad=False)
        else:
            input_batch = Variable(input_batch, requires_grad=False)

        start = time.time()
        output_batch, _ = model(input_batch)
        elapsed_time = time.time() - start
        avrtime += elapsed_time

        output_batch = (output_batch.data + 0.5) * 255  # change into pixel domain
        output_batch = output_batch.cpu().numpy()
        export(args.scale, model.name, output_batch, imgname[0])
        cnt += 1

    avrtime = avrtime / cnt
    print('Average time: %f\n' % avrtime)
    print('..Finish..')

if __name__ == '__main__':
    main()

