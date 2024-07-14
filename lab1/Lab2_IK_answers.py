import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    joint_parent = meta_data.joint_parent
    joint_offset = [meta_data.joint_initial_position[i]-meta_data.joint_initial_position[joint_parent[i]] for i in range(len(joint_positions))]
    joint_offset[0] = np.array([0.,0.,0.])
    joint_ik_path,_,_,_ = meta_data.get_path_from_root_to_end()
    # 用于迭代计算IK链条上各个关节的旋转
    local_rotation = [R.from_quat(joint_orientations[joint_parent[i]]).inv()*R.from_quat(joint_orientations[i]) for i in range(len(joint_orientations))]
    local_rotation[0] = R.from_quat(joint_orientations[0])

    # 梯度下降的方法
    joint_offset_t = [torch.tensor(data) for data in joint_offset]
    joint_positions_t = [torch.tensor(data) for data in joint_positions]
    joint_orientations_t = [torch.tensor(R.from_quat(data).as_matrix(), requires_grad=True) for data in joint_orientations]
    local_rotation_t = [torch.tensor(data.as_matrix(), requires_grad=True) for data in local_rotation]
    target_pose_t = torch.tensor(target_pose)

    epoch = 60000
    alpha = 0.01
    for _ in range(epoch):
        for j in range(len(joint_ik_path)):
            # 更新链上结点的位置
            a = chain_current = joint_ik_path[j]
            b = chain_parent = joint_ik_path[j-1]
            if j == 0:
                local_rotation_t[a] = local_rotation_t[a]
                joint_positions_t[a] = joint_positions_t[a]
            elif b == joint_parent[a]:
                joint_orientations_t[a] = joint_orientations_t[b] @ local_rotation_t[a]
                joint_positions_t[a] = joint_positions_t[b] + joint_offset_t[a] @ torch.transpose(joint_orientations_t[b],0,1)
            # else:
            #     joint_orientations_t[a] = joint_orientations_t[b] @ torch.transpose(local_rotation_t[b],0,1)
            #     joint_positions_t[a] = joint_positions_t[b] + (-joint_offset_t[b]) @ torch.transpose(joint_orientations_t[b],0,1)
        
        optimize_target = torch.norm(joint_positions_t[joint_ik_path[-1]]-target_pose_t)
        optimize_target.backward()
        for num in joint_ik_path:
            if local_rotation_t[num].grad is not None:
                tmp = local_rotation_t[num] - alpha * local_rotation_t[num].grad
                local_rotation_t[num] = torch.tensor(tmp,requires_grad=True)
    
    for j in range(len(joint_ik_path)):
        a = chain_current = joint_ik_path[j]
        b = chain_parent = joint_ik_path[j-1]
        if j == 0:
            local_rotation[a] = R.from_matrix(local_rotation_t[a].detach().numpy())
            joint_positions[a] = joint_positions[a]
        elif b == joint_parent[a]:
            joint_orientations[a] = (R.from_quat(joint_orientations[b]) * R.from_matrix(local_rotation_t[a].detach().numpy())).as_quat()
            joint_positions[a] = joint_positions[b] + joint_offset[a] * np.asmatrix(R.from_quat(joint_orientations[b]).as_matrix()).transpose()
        # else:
        #     joint_orientations[a] = (R.from_quat(joint_orientations[b]) * R.from_matrix(local_rotation_t[b].detach().numpy()).inv()).as_quat()
        #     joint_positions[a] = joint_positions[b] + (-joint_offset[b]) * np.asmatrix(R.from_quat(joint_orientations[b]).as_matrix()).transpose()
    
    # 我们获得了链条上每个关节的Orientation和Position, 然后我们只需要更新非链上结点的位置
    ik_path_set = set(joint_ik_path)
    for i in range(len(joint_positions)):
        if i in ik_path_set:
            joint_orientations[i] = joint_orientations[i]
            # joint_orientations[i] = R.from_matrix(joint_orientations[i].detach().numpy()).as_quat()
        else:
            joint_orientations[i] = (R.from_quat(joint_orientations[joint_parent[i]]) * local_rotation[i]).as_quat()
            joint_positions[i] = joint_positions[joint_parent[i]] + joint_offset[i] * np.asmatrix(
                R.from_quat(joint_orientations[joint_parent[i]]).as_matrix()).transpose()

    return joint_positions, joint_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations