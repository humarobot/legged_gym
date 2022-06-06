import unittest
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger,helpers
import torch
import time 
import math

class TrajTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.device = 'cuda'
        self.num_envs = 512
        self.num_actions = 12

    def test_forward_kinematics(self):
        self.dof_pos = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.side_sign = torch.zeros(self.num_envs , 4, dtype=torch.float, device=self.device, requires_grad=False)    
        self.side_sign[:,[1,3]] -= 1.
        self.side_sign[:,[0,2]] += 1.  
        s1 = torch.sin(self.dof_pos[:,[0,3,6,9]])
        s2 = torch.sin(self.dof_pos[:,[1,4,7,10]])
        s3 = torch.sin(self.dof_pos[:,[2,5,8,11]])
        c1 = torch.cos(self.dof_pos[:,[0,3,6,9]])
        c2 = torch.cos(self.dof_pos[:,[1,4,7,10]])
        c3 = torch.cos(self.dof_pos[:,[2,5,8,11]])
        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3
        l1 = 0.119
        l2 = 0.3
        l3 = 0.308
        dof_foot_x = l3 * s23 + l2 * s2
        dof_foot_y = (l1) * self.side_sign * c1 + l3 * (s1 * c23) + l2 * c2 * s1
        dof_foot_z = (l1) * self.side_sign * s1 - l3 * (c1 * c23) - l2 * c1 * c2

        print(' '.join('{:<8.4f}'.format(x) for x in dof_foot_x[0]),' '.join('{:<8.4f}'.format(y) for y in dof_foot_y[0]),' '.join('{:<8.4f}'.format(z) for z in dof_foot_z[0]))


    # def test_foot_traj(self):
    #     self.t_reset = torch.zeros(self.num_envs, 1 ,dtype=torch.float, device=self.device, requires_grad=False)
    #     self.t_now = torch.zeros(self.num_envs, 1 ,dtype=torch.float, device=self.device, requires_grad=False)
    #     self.ref_foot_x = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
    #     self.ref_foot_y = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
    #     self.ref_foot_z = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
    #     self.init_foot_phase = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
    #     self.ref_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
    #     self.init_foot_phase[:,[1,3]] = 0.5
    #     self.ref_foot_y[:,[1,3]] = -0.119
    #     self.ref_foot_y[:,[0,2]] = 0.119
        

    #     freq = 1.0
    #     T = 1.0 / freq
    #     dt = 0.02
    #     l1 = 0.119
    #     l2 = 0.3
    #     l3 = 0.308
        

    #     while(1):
    #         k = 4*((self.init_foot_phase*T + self.t_now.repeat(1,4)) % T) / T
    #         up_buf = (k<1).nonzero(as_tuple=True)
    #         down_buf = ((k>=1) & (k<2)).nonzero(as_tuple=True)
    #         stance_buf = (k>=2).nonzero(as_tuple=True)
    #         self.ref_foot_z[up_buf] = 0.5*(-2*k[up_buf]**3+3*k[up_buf]**2)-0.5
    #         self.ref_foot_z[down_buf] = 0.5*(2*k[down_buf]**3-9*k[down_buf]**2+12*k[down_buf]-4)-0.5
    #         self.ref_foot_z[stance_buf] = -0.5
    #         # print(self.t_now[0],self.ref_foot_z[0])

    #         self.ref_foot_x += -0.3
    #         self.ref_foot_y += -0.3
    #         self.ref_foot_z += -0.3

    #         length = torch.sqrt(self.ref_foot_x**2+self.ref_foot_y**2+self.ref_foot_z**2)
    #         max = math.sqrt(l1**2+(l2+l3)**2)-0.05
    #         min = l1+0.05
    #         longer_index = (length > max).nonzero(as_tuple=True) 
    #         shorter_index = (length < min).nonzero(as_tuple=True)
    #         self.ref_foot_x[longer_index] *= (max/length[longer_index])
    #         self.ref_foot_y[longer_index] *= (max/length[longer_index])
    #         self.ref_foot_z[longer_index] *= (max/length[longer_index])
    #         self.ref_foot_x[shorter_index] *= (min/length[shorter_index])
    #         self.ref_foot_y[shorter_index] *= (min/length[shorter_index])
    #         self.ref_foot_z[shorter_index] *= (min/length[shorter_index])

            

    #         H =torch.sqrt(self.ref_foot_y**2+self.ref_foot_z**2-l1**2)
    #         self.ref_actions[:,[3,9]] = torch.atan(l1/H[:,[1,3]])+torch.atan2(self.ref_foot_y[:,[1,3]],-self.ref_foot_z[:,[1,3]])
    #         self.ref_actions[:,[0,6]] = torch.atan(H[:,[0,2]]/l1)-torch.atan2(-self.ref_foot_z[:,[0,2]],self.ref_foot_y[:,[0,2]])
    #         m1 = (self.ref_foot_x**2+H**2+l2**2-l3**2)/(2*torch.sqrt(self.ref_foot_x**2+H**2)*l2) 
    #         self.ref_actions[:,[1,4,7,10]] = torch.atan(-self.ref_foot_x/H)+torch.acos(m1)
    #         m2 = (self.ref_foot_x**2+H**2-l2**2-l3**2)/(2*l2*l3)
    #         self.ref_actions[:,[2,5,8,11]] = -torch.acos(m2)
    #         # print(self.ref_actions[0].tolist())
    #         print(''.join('{:<+8.2f}'.format(k) for k in self.ref_actions[0]))
    #         self.t_now += dt
    #         time.sleep(0.02)


if __name__=="__main__":
    unittest.main()