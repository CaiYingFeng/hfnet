# import numpy as np
# import argparse
# import yaml
# import logging
# from pathlib import Path
# from tqdm import tqdm
# from pprint import pformat

# logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)
# from hfnet.models import get_model  # noqa: E402
# from hfnet.datasets import get_dataset  # noqa: E402
# from hfnet.utils import tools  # noqa: E402
# from hfnet.settings import EXPER_PATH, DATA_PATH  # noqa: E402


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('config', type=str)
#     parser.add_argument('export_name', type=str)
#     parser.add_argument('--keys', type=str, default='*')
#     parser.add_argument('--exper_name', type=str)
#     parser.add_argument('--as_dataset', action='store_true')
#     args = parser.parse_args()

#     export_name = args.export_name
#     exper_name = args.exper_name
#     with open(args.config, 'r') as f:
#         config = yaml.load(f)
#     keys = '*' if args.keys == '*' else args.keys.split(',')

#     if args.as_dataset:
#         base_dir = Path(DATA_PATH, export_name)
#     else:
#         base_dir = Path(EXPER_PATH, 'exports')
#         base_dir = Path(base_dir, ((exper_name+'/') if exper_name else '') + export_name)
#     base_dir.mkdir(parents=True, exist_ok=True)

#     if exper_name:
#         # Update only the model config (not the dataset)
#         with open(Path(EXPER_PATH, exper_name, 'config.yaml'), 'r') as f:
#             config['model'] = tools.dict_update(
#                 yaml.load(f)['model'], config.get('model', {}))
#         checkpoint_path = Path(EXPER_PATH, exper_name)
#         if config.get('weights', None):
#             checkpoint_path = Path(checkpoint_path, config['weights'])
#     else:
#         if config.get('weights', None):
#             checkpoint_path = Path(DATA_PATH, 'weights', config['weights'])
#         else:
#             checkpoint_path = None
#             logging.info('No weights provided.')
#     logging.info(f'Starting export with configuration:\n{pformat(config)}')

#     with get_model(config['model']['name'])(
#             data_shape={'image': [None, None, None, config['model']['image_channels']]},
#             **config['model']) as net:
#         if checkpoint_path is not None:
#             net.load(str(checkpoint_path))
#         dataset = get_dataset(config['data']['name'])(**config['data'])
#         test_set = dataset.get_test_set()

#         for data in tqdm(test_set):
#             predictions = net.predict(data, keys=keys)
#             predictions['input_shape'] = data['image'].shape
#             name = data['name'].decode('utf-8')
#             Path(base_dir, Path(name).parent).mkdir(parents=True, exist_ok=True)
#             np.savez(Path(base_dir, '{}.npz'.format(name)), **predictions)

import sys
import os
sys.path.append("./")
import numpy as np
import argparse
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from pprint import pformat
import h5py
import os
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
from hfnet.models import get_model  # noqa: E402
from hfnet.datasets import get_dataset  # noqa: E402
from hfnet.utils import tools  # noqa: E402
from hfnet.settings import EXPER_PATH, DATA_PATH  # noqa: E402

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parser = argparse.ArgumentParser()
    parser.add_argument('method',type=str)

    # parser.add_argument('config', type=str)
    # parser.add_argument('export_name', type=str)
    # parser.add_argument('--keys', type=str, default='*')
    # parser.add_argument('--exper_name', type=str)
    parser.add_argument('--as_dataset', action='store_true')
    args = parser.parse_args()

    #method='superpoint_queries'

    method=args.method
    if method=='hfnet_db':
        export_name = 'aachen'
        exper_name = 'hfnet'
        myconfig='hfnet/configs/hfnet_export_aachen_db.yaml'
        mykeys='keypoints,scores,local_descriptor_map,global_descriptor'
    if method=='hfnet_queries':
        export_name = 'aachen'
        exper_name = 'hfnet'
        myconfig='hfnet/configs/hfnet_export_aachen_queries.yaml'
        mykeys='keypoints,scores,local_descriptor_map,global_descriptor'
    if method=='superpoint_db':
        export_name = 'superpoint/aachen'
        exper_name=''
        myconfig='hfnet/configs/superpoint_export_aachen_db.yaml'
        mykeys='keypoints,scores,local_descriptor_map'
    if method=='superpoint_queries':
        export_name = 'superpoint/aachen'
        exper_name=''
        myconfig='hfnet/configs/superpoint_export_aachen_queries.yaml'
        mykeys='keypoints,scores,local_descriptor_map'
    if method=='netvlad':
        export_name = 'netvlad/aachen'
        exper_name=''
        myconfig='hfnet/configs/netvlad_export_aachen.yaml'
        mykeys='global_descriptor'
    if method=='superpoint':
        export_name = 'google_landmarks/superpoint_predictions'
        exper_name=''
        myconfig='hfnet/configs/superpoint_export_distill.yaml'
        mykeys='local_descriptor_map,dense_scores'

    

    

    # export_name = args.export_name
    # exper_name = args.exper_name
    with open(myconfig, 'r') as f:
        config = yaml.load(f)
    keys = '*' if mykeys == '*' else mykeys.split(',')

    if args.as_dataset:
        base_dir = Path(DATA_PATH, export_name)
    else:
        base_dir = Path(EXPER_PATH, 'exports')
        base_dir = Path(base_dir, ((exper_name+'/') if exper_name else '') + export_name)
    base_dir.mkdir(parents=True, exist_ok=True)
   
    if exper_name:
        # Update only the model config (not the dataset)
        with open(Path(EXPER_PATH, exper_name, 'config.yaml'), 'r') as f:
            print(Path(EXPER_PATH, exper_name, 'config.yaml'))
            config['model'] = tools.dict_update(
                yaml.load(f)['model'], config.get('model', {}))
            print(config['model'])
        checkpoint_path = Path(EXPER_PATH, exper_name)
        print(checkpoint_path)
        if config.get('weights', None):
            checkpoint_path = Path(checkpoint_path, config['weights'])
        print(checkpoint_path) 
    else:
        if config.get('weights', None):
            checkpoint_path = Path(DATA_PATH, 'weights', config['weights'])
        else:
            checkpoint_path = None
            logging.info('No weights provided.')
    logging.info(f'Starting export with configuration:\n{pformat(config)}')

    with get_model(config['model']['name'])(
            data_shape={'image': [None, None, None, config['model']['image_channels']]},
            **config['model']) as net:
       
        if checkpoint_path is not None:
            net.load(str(checkpoint_path))
        
        dataset = get_dataset(config['data']['name'])(**config['data'])
       
        test_set = dataset.get_test_set()

        # feature_file = h5py.File(Path(base_dir, 'all_1.h5'), 'a')#生成h5所需

        for data in tqdm(test_set):
            # print (data)
            # break
            # print(name)

            predictions = net.predict(data, keys=keys)
            predictions['input_shape'] = data['image'].shape
            name = data['name'].decode('utf-8')#name: db/1606404423.00735318
            Path(base_dir, Path(name).parent).mkdir(parents=True, exist_ok=True)
            np.savez(Path(base_dir, '{}.npz'.format(name)), **predictions)

            ###########################
            ###生成pairs需要的h5文件####
            ###########################

            # if(name.split('.',-1)[-1]=='jpg'):
            #     name+='.png'
            # else:
            #     name+='.jpg'

            # grp=feature_file.create_group(name)
            # grp.create_dataset('global_descriptor',data=predictions['global_descriptor'])
            # grp.create_dataset('input_shape',data=predictions['input_shape'])
            # break

            ##########################
            ##########################
            ##########################
        
        # feature_file.close()
    
