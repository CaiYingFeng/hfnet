import cv2
import numpy as np
from pathlib import Path
import yaml
from PIL import Image
from hfnet.datasets.aachen import Aachen
from hfnet.evaluation.localization import Localization
from hfnet.evaluation.utils.db_management import read_query_list
from hfnet.evaluation.loaders import export_loader,export_loader_db
from hfnet.settings import DATA_PATH,EXPER_PATH

from notebooks.utils import plot_matches
from notebooks.utils import plot_keypoints
import operator
from functools import reduce
from pyquaternion import Quaternion

config_global = {
    'db_name': 'globaldb_netvlad.pkl',
    'experiment': 'netvlad/aachen',
    'predictor': export_loader, 
    'has_keypoints': False, 
    'has_descriptors': False, 
    'pca_dim': 4096,
    'num_prior': 10,
}
config_local = {
    'db_name': 'localdb_superpoint_BA.pkl',
    'experiment': 'superpoint/aachen',
    'predictor': export_loader,
    'has_keypoints': True,
    'has_descriptors': True,
#    'do_nms': True,
#    'nms_thresh': 4,
    'num_features': 2000,
    'ratio_thresh': 0.9,
}

# config_local = {
#         'db_name': 'localdb_hfnet.pkl',
#         'experiment':  'hfnet/aachen',
#         'predictor_db': export_loader_db,
#         'predictor': export_loader,
#         'has_keypoints': True,
#         'has_descriptors': True,
#         # 'do_nms': True,
#         # 'nms_thresh': 4,
#         'num_features': 2000,
#         'ratio_thresh': 0.9,
#     }

# config_local = {
#     'db_name': 'localdb_sift_test.pkl',
#     'colmap_db': 'cam03.db',
#     'ratio_thresh': 0.7,
#     'root': False,
#     'fast_matching': False,
# }
model = 'cam03_BA'

config_pose = {
    'reproj_error': 10,
    'min_inliers': 12,
}
config = {'global': config_global, 'local': config_local, 'pose': config_pose}
loc = Localization('aachen', model, config)
queries = read_query_list(Path(loc.base_path, 'cam03.txt'))


# eval_file = 'aachen/cam03.yaml'
# with open(Path(EXPER_PATH, 'eval', eval_file), 'r') as f:
#     failures = yaml.load(f)['metrics']['failure']
#print(failures)
# queries = [queries[f] for f in failures]  # failures
# queries = [queries[i] for i in range(len(failures))]
#print(len(queries))
# queries = [queries[i] for i in range(len(queries)) if i not in set(failures)]  # success
# print(len(queries))
# print(queries[0])
# print(queries[1])

# np.random.RandomState(0).shuffle(queries)

q_name=['query/1606405726.20872613.png']
# query_dataset = Aachen(**{'resize_max': 1920,
#                           'image_names': q_name})
query_dataset = Aachen(**{'resize_max': 1920,
                          'image_names': [q.name for q in queries]})

def get_image(name):
    path = Path(DATA_PATH, query_dataset.dataset_folder,name)
    return cv2.imread(path.as_posix())[..., ::-1]
def get_image2(name):
    path = Path(DATA_PATH, query_dataset.dataset_folder,'db',name)
    return cv2.imread(path.as_posix())[..., ::-1]

query_iter = query_dataset.get_test_set()
# myname=['query/1594465419.306.jpg','query/1594465503.420.jpg','query/1594466227.035.jpg']
for i, query_info, query_data in zip(range(0,len(queries)), queries, query_iter):
    
    if query_info.name in q_name :
        # print(1)
        results, debug = loc.localize(query_info, query_data, debug=True)

        s = f'{i} {"Success" if results.success else "Failure"}, inliers {results.num_inliers:^4}, ' \
            + f'ratio {results.inlier_ratio:.3f}, landmarks {len(debug["matching"]["lm_frames"]):>4}, ' \
            + f'spl {debug["index_success"]:>2}, places {[len(p) for p in debug["places"]]:}, ' \
            + f'pos {[f"{n:.1f}" for n in results.T[:3, 3]]}'
        query_T_w = np.linalg.inv(results.T)
        qvec_nvm = list(Quaternion(matrix=query_T_w))
        pos_nvm = query_T_w[:3, 3].tolist()
        print("QUERT IMAGE---"+query_info.name)
        path_query=Path(DATA_PATH,'aachen/image',query_info.name)
        # display(Image.open(path_query).resize((300,300)))

        p=reduce(operator.add, debug["places"])
        print("TOP K ")
        print(p)
        for k in range(0,len(p)):
            p[k]=loc.images[p[k]].name
        print(p)
   

        for pname in p:
            path_topk=Path(DATA_PATH, 'aachen/image/db',pname)

            print(pname)

        sorted_frames, counts = np.unique(
            [debug['matching']['lm_frames'][m2] for m1, m2 in debug['matches'][debug['inliers']]],
            return_counts=True)

        best_id = sorted_frames[np.argmax(counts)]


        query_image = get_image(query_info.name)

        best_image = get_image2(loc.images[best_id].name)

    
        print("FINAL_IMAGE:"+loc.images[best_id].name)
        path_final=Path(DATA_PATH,'aachen/image/db',loc.images[best_id].name)
        # display(Image.open(path_final).resize((300,300)))

        best_matches_inliers = [(m1, debug['matching']['lm_indices'][m2]) 
                                for m1, m2 in debug['matches'][debug['inliers']] 
                                if debug['matching']['lm_frames'][m2] == best_id]

        best_matches_outliers = [(m1, debug['matching']['lm_indices'][m2])
                                for i, (m1, m2) in enumerate(debug['matches']) 
                                if debug['matching']['lm_frames'][m2] == best_id
                                and i not in debug['inliers']]

        all_matches=best_matches_inliers+best_matches_outliers

        plot_matches(
            query_image, debug['query_item'].keypoints,
            best_image, loc.local_db[best_id].keypoints,
            np.array(best_matches_inliers), color=(0, 1., 0),
            dpi=100, ylabel=str(i), thickness=1.)
        plot_matches(
            query_image, debug['query_item'].keypoints,
            best_image, loc.local_db[best_id].keypoints,
#             np.array(best_matches_inliers), color=(1., 0., 0),
            np.array(best_matches_outliers), color=(1., 0, 0),       
            dpi=100, ylabel=str(i), thickness=1.)
        plot_keypoints(
            query_image, debug['query_item'].keypoints,
            best_image, loc.local_db[best_id].keypoints,
            np.array(best_matches_inliers), color=(0, 1., 0),
            dpi=100, ylabel=str(i), thickness=1.)
        plot_keypoints(
            query_image, debug['query_item'].keypoints,
            best_image, loc.local_db[best_id].keypoints,
            #np.array(best_matches_inliers), color=(0, 1., 0),
            np.array(best_matches_outliers), color=(1., 0, 0),       
            dpi=100, ylabel=str(i), thickness=1.)
        print(qvec_nvm,pos_nvm)

        print("###########################################################################################")