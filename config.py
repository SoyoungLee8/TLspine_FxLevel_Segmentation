import argparse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu_num', type=str, default='0')
    parser.add_argument('--DATA_DIR', type=str, default='')
    parser.add_argument('--train_img', type=str, default='', help='A train x-ray floder name (PNG, 1024x1024 size)')
    parser.add_argument('--train_fx', type=str, default='', help='A train fx segmentation map floder name (PNG, 1024x1024 size)')
    parser.add_argument('--train_level', type=str, default='', help='A train level segmentation map floder name (PNG, 1024x1024 size)')
    parser.add_argument('--test_img', type=str, default='', help='A test x-ray floder name (PNG, 1024x1024 size)')
    parser.add_argument('--test_fx', type=str, default='', help='A test fx segmentation map floder name (PNG, 1024x1024 size)')
    parser.add_argument('--test_level', type=str, default='', help='A test level segmentation map floder name (PNG, 1024x1024 size)')
    
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--experiment_name', type=str, default='experiment_1')

    return parser.parse_args()
