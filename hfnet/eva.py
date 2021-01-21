import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import os
from pathlib import Path
import argparse
import subprocess
import logging

def eva(model, eval_name, local_method, global_method, queries, output_dir):
    
    print('Running evaluate...')
    

    cmd = [
        'python','hfnet/evaluate_aachen.py',

        '--model', str(model),
        '--eval_name', str(eval_name),
        '--local_method',str(local_method),
        '--global_method',str(global_method),
        '--queries',str(queries),
        '--output_dir',str(output_dir)
        ]

    print(cmd)
    ret = subprocess.call(cmd)
    if ret != 0:
        logging.warning('Problem with evaluate, exiting.')
        exit(ret)

    print('Finished evaluate...')

def w2c(eval_name, output_dir):
#从hfnet的pose里面提取位姿
#输入：name qw,qx,qy,qz,tx,ty,tz   :c2w
#输出：time tx ty tz qx qy qz qw   :w2c
    cam=eval_name
    if os.path.exists(output_dir+'/'+cam+'/estimate1.txt'):
            os.remove(output_dir+'/'+cam+'/estimate1.txt')
    i_path=output_dir+'/'+cam+'/estimate1.txt'
    f=open(output_dir+'/'+cam+'/'+cam+'_poses.txt')
    f_dof=list(f)
    f.close
    for i in range(0,len(f_dof)):
        str_dof=f_dof[i].split(' ',-1)             
        with open(i_path, 'a') as f:
            line=str_dof[0].strip(".png")+' '          
                    
            line+=str_dof[5]+' '
            line+=str_dof[6]+' '
            line+=str_dof[7].strip("\n")+' ' 
                    
                    
            line+=str_dof[2]+' '
            line+=str_dof[3]+' '
            line+=str_dof[4]+' '
            line+=str_dof[1]+' '
                    
                    
            
            f.write(line+'\n')

    #f = open("/media/autolab/disk_3T/caiyingfeng/stamped_groundtruth.txt","r")#w2c:time tx ty tz qx qy qz qw 
    f = open(output_dir+'/'+cam+'/estimate1.txt','r')#w2c:time tx ty tz qx qy qz qw 


    f_dof=list(f)
    f_dof.sort()#因为groundtruth是三个相机按序来的，所以不估计值也不sort
    f.close
    #i_path=Path('/media/autolab/disk_3T/caiyingfeng/rpg_trajectory_evaluation/eva',f'stamped_groundtruth.txt')#要保存的c2w:time tx ty tz qx qy qz qw 
    i_path=Path(output_dir+'/'+cam,f'estimate.txt')#要保存的c2w:time tx ty tz qx qy qz qw 
    for i in range(0,len(f_dof)):               
        str_dof=f_dof[i].split(' ',-1)
        with open(i_path, 'a') as f:
            qw = float(str_dof[7])        
            qx = float(str_dof[4])
            qy = float(str_dof[5])
            qz = float(str_dof[6])
            
            tx = float(str_dof[1])
            ty = float(str_dof[2])
            tz = float(str_dof[3])
            #print(qw)
    #         r = Quaternion(qw, qx, qy, qz)
    #         rotation = r.rotation_matrix
            r = R.from_quat([qx,qy,qz,qw])
            rotation = r.as_matrix()
            
            translation = np.asarray([tx,ty,tz])
            xyz = -np.dot(np.linalg.inv(rotation),translation)
            xyz = np.reshape(xyz,(1,3))
    #         print(xyz)
    #         print(xyz[0][1])
            
            r = R.from_matrix(np.linalg.inv(rotation))
            r2=r.as_quat()
            
            #print(res)

            line=str_dof[0]+" "#time

            #tx ty tz
            # line+=(xyz[0][0]+368516).__str__()+' '   
            # line+=(xyz[0][1]+3459036).__str__()+' '
            # line+=(xyz[0][2]+15).__str__()+' '
            line+=(xyz[0][0]).__str__()+' '   
            line+=(xyz[0][1]).__str__()+' '
            line+=(xyz[0][2]).__str__()+' '


            #qx qy qz qw
            line+=r2[0].__str__()+' '
            line+=r2[1].__str__()+' '
            line+=r2[2].__str__()+' '
            line+=r2[3].__str__()
            #print(line)


            #break
    
            f.write(line+'\n')

def evo(output_dir,eval_name,gt_txt):

    estimate_txt=output_dir+'/'+eval_name+'estimate.txt'   

    cmd = [
        'evo','tum',
        '--model', str(model),
        str(estimate_txt),
        str(gt_txt),
        
        '-p','--output_dir==xy'
        ]

    print(cmd)
    ret = subprocess.call(cmd)
    if ret != 0:
        logging.warning('Problem with evo, exiting.')
        exit(ret)
    
if __name__ == "__main__":
    
    opts = argparse.ArgumentParser("This script is used to evaluate.")
    opts.add_argument("--model", default='HL_model')#sfm模型
    opts.add_argument("--eval_name", default='HL_sp_cam03')#评估输出的txt名字
    opts.add_argument("--local_method", default='superpoint')#局部特征匹配方法
    opts.add_argument("--global_method", default='netvlad')#全局检索方法
    opts.add_argument("--queries", default='cam03')#检索图片的txtlist
    opts.add_argument("--output_dir", default='/media/autolab/disk_4T/cyf/localization/out/eval/aachen')#hfnet输出结果位置为：output_dir+eval_name
    opts.add_argument("--gt_txt", default='/media/autolab/disk_4T/cyf/localization/out/eval/aachen/gt/cam03.txt')#真值位姿位置

    opts = opts.parse_args()
    model=opts.model
    eval_name=opts.eval_name
    local_method=opts.local_method
    global_method=opts.global_method
    queries=opts.queries
    output_dir=opts.output_dir
    gt_txt=opts.gt_txt

    eva(model, eval_name, local_method, global_method, queries, output_dir)
    w2c(eval_name, output_dir)
    # evo(output_dir, eval_name, gt_txt)
    # evo_ape tum cam03.txt estimate.txt -p --plot_mode=xy