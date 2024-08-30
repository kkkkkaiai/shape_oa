import torch
import time, os
import glob
import numpy as np
import trimesh
import skimage

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CUR_DIR, '../../../resources/robots/obstacles')

class BPSDF():
    def __init__(self, n_func,domain_min,domain_max,device):
        self.n_func = n_func
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device    
        self.model_path = os.path.join(CUR_DIR, 'models')
        
    def binomial_coefficient(self, n, k):
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

    def build_bernstein_t(self,t, use_derivative=False):
        # bernstein_t is a function that returns the Bernstein polynomial of degree n evaluated at t
        # t is normalized to [0,1]
        t =torch.clamp(t, min=1e-4, max=1-1e-4)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        comb = self.binomial_coefficient(torch.tensor(n, device=self.device), i)
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i
        if not use_derivative:
            return phi.float(),None
        else:
            dphi = -comb * (n - i) * (1 - t).unsqueeze(-1) ** (n - i - 1) * t.unsqueeze(-1) ** i + comb * i * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** (i - 1)
            dphi = torch.clamp(dphi, min=-1e4, max=1e4)
            return phi.float(),dphi.float()

    def build_basis_function_from_points(self, p, use_derivative=False):
        N = len(p)
        p = ((p - self.domain_min)/(self.domain_max-self.domain_min)).reshape(-1)
        phi,d_phi = self.build_bernstein_t(p,use_derivative) 
        phi = phi.reshape(N,3,self.n_func)
        phi_x = phi[:,0,:]
        phi_y = phi[:,1,:]
        phi_z = phi[:,2,:]
        phi_xy = torch.einsum("ij,ik->ijk",phi_x,phi_y).view(-1,self.n_func**2)
        phi_xyz = torch.einsum("ij,ik->ijk",phi_xy,phi_z).view(-1,self.n_func**3)
        if use_derivative ==False:
            return phi_xyz,None
        else:
            d_phi = d_phi.reshape(N,3,self.n_func)
            d_phi_x_1D= d_phi[:,0,:]
            d_phi_y_1D = d_phi[:,1,:]
            d_phi_z_1D = d_phi[:,2,:]
            d_phi_x = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",d_phi_x_1D,phi_y).view(-1,self.n_func**2),phi_z).view(-1,self.n_func**3)
            d_phi_y = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",phi_x,d_phi_y_1D).view(-1,self.n_func**2),phi_z).view(-1,self.n_func**3)
            d_phi_z = torch.einsum("ij,ik->ijk",phi_xy,d_phi_z_1D).view(-1,self.n_func**3)
            d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1),d_phi_y.unsqueeze(-1),d_phi_z.unsqueeze(-1)),dim=-1)
            return phi_xyz,d_phi_xyz
        
    def sdf_to_mesh(self, model, nbData,use_derivative=False):
        verts_list, faces_list, mesh_name_list = [], [], []
        for i, k in enumerate(model.keys()):
            mesh_dict = model[k]
            mesh_name = mesh_dict['mesh_name']
            print(f'{mesh_name}')
            mesh_name_list.append(mesh_name)
            weights = mesh_dict['weights'].to(self.device)

            domain = torch.linspace(self.domain_min,self.domain_max,nbData).to(self.device)
            grid_x, grid_y, grid_z= torch.meshgrid(domain,domain,domain)
            grid_x, grid_y, grid_z = grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)
            p = torch.cat([grid_x, grid_y, grid_z],dim=1).float().to(self.device)   

            # split data to deal with memory issues
            p_split = torch.split(p, 10000, dim=0)
            d =[]
            
            for p_s in p_split:
                phi_p,d_phi_p = self.build_basis_function_from_points(p_s,use_derivative)
                d_s = torch.matmul(phi_p,weights)
                d.append(d_s)
            d = torch.cat(d,dim=0)
            

            start_time = time.time()
            test_data = torch.Tensor([[0,0,0]]).to(self.device)
            phi_test,_ = self.build_basis_function_from_points(test_data,use_derivative)
            d_test = torch.matmul(phi_test,weights)

            verts, faces, normals, values = skimage.measure.marching_cubes(
                d.view(nbData,nbData,nbData).detach().cpu().numpy(), level=0.0, spacing=np.array([(self.domain_max-self.domain_min)/nbData] * 3)
            )
            verts = verts - [self.domain_max,self.domain_max,self.domain_max] # 修正坐标系
            verts_list.append(verts)
            faces_list.append(faces)
        return verts_list, faces_list,mesh_name_list

    def create_surface_mesh(self,model, nbData,vis =False, save_mesh_name=None):
        verts_list, faces_list,mesh_name_list = self.sdf_to_mesh(model, nbData)
        for verts, faces,mesh_name in zip(verts_list, faces_list,mesh_name_list):
            rec_mesh = trimesh.Trimesh(verts,faces)
            # add a coordiname frame in recmesh
            coord_mesh = trimesh.creation.axis(origin_size=0.0001,axis_radius=0.01,axis_length=1.0)
            rec_mesh = rec_mesh + coord_mesh

            if vis:
                rec_mesh.show()
            if save_mesh_name != None:
                save_path = os.path.join(CUR_DIR,"output_meshes")
                if os.path.exists(save_path) is False:
                    os.mkdir(save_path)
                trimesh.exchange.export.export_mesh(rec_mesh, os.path.join(save_path,f"{save_mesh_name}_{mesh_name}.stl"))

def quat_to_transform_matrix(quat, pos=None, if_tensor=False, device='cuda'):
    x, y, z, w = quat
    if pos is not None:
        pos_x, pos_y, pos_z = pos
    else:
        pos_x, pos_y, pos_z = 0, 0, 0
    if if_tensor:
        transform_matrix = torch.tensor([[1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w, pos_x],
                                         [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w, pos_y],
                                         [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2, pos_z],
                                         [0, 0, 0, 1]], device=device)
    else:
        transform_matrix = np.array([[1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w, pos_x],
                                    [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w, pos_y],
                                    [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2, pos_z],
                                    [0, 0, 0, 1]])
    return transform_matrix



class DistanceCheck():
    def __init__(self, device='cuda'):
        self.domain_min = -2.0
        self.domian_max = 2.0
        self.obs_n_func = 12
        self.robot_n_func = 24
        self.device = device
        self.obs_bp_sdf = BPSDF(self.obs_n_func, self.domain_min, self.domian_max, self.device)
        self.robot_bp_sdf = BPSDF(self.robot_n_func, self.domain_min, self.domian_max, self.device)

        files = glob.glob(os.path.join(MODEL_DIR, '*.pt'))
            
        # assign path
        for i in range(len(files)):
            # print(files[i])
            if 'cylinder' in files[i]:
                cylinder_path = files[i]
            if 'rect' in files[i]:
                rect_path = files[i]
            if 'triangle' in files[i]:
                triangle_path = files[i]
            if 'pentagon' in files[i]:
                pentagon_path = files[i]
            if 'swerve' in files[i]:
                robot_path = files[i]
        
        self.cylinder_model = torch.load(cylinder_path)
        self.rect_model = torch.load(rect_path)
        self.triangle_model = torch.load(triangle_path)
        self.pentagon_model = torch.load(pentagon_path)
        self.robot_model = torch.load(robot_path)

    def check(self, obs, robot, reverse=False, if_tensor=False):
        '''
        obs: each obstacle is a dict with keys: 'name', 'position', 'orientation'
        orientation is a quaternion
        '''
        robot_pos = robot['position']
        robot_ori = robot['orientation']

        name = obs['name']
        obs_pos = obs['position']
        obs_ori = obs['orientation']


        robot_trans_matrix = quat_to_transform_matrix(robot_ori, robot_pos, if_tensor=if_tensor)
        obs_trans_matrix = quat_to_transform_matrix(obs_ori, obs_pos, if_tensor=if_tensor)

        if 'cylinder' in name:
            model = self.cylinder_model
        if 'rect' in name:
            model = self.rect_model
        if 'triangle' in name:
            model = self.triangle_model
        if 'pentagon' in name:
            model = self.pentagon_model

        # calculate the relative transform matrix based on the obs coordinate
        if not reverse:
            if if_tensor:
                relative_matrix = torch.matmul(torch.inverse(obs_trans_matrix), robot_trans_matrix)
                position = torch.tensor([relative_matrix[0:3, 3].tolist()]).to(self.device)
            else:
                relative_matrix = np.dot(np.linalg.inv(obs_trans_matrix), robot_trans_matrix)
                position = torch.tensor([relative_matrix[0:3, 3]]).to(self.device)
            phi_p, d_phi_p = self.obs_bp_sdf.build_basis_function_from_points(position, True)
        else:
            model = self.robot_model
            if if_tensor:
                relative_matrix = torch.matmul(torch.inverse(robot_trans_matrix), obs_trans_matrix)
                position = torch.tensor([relative_matrix[0:3, 3].tolist()]).to(self.device)
            else:
                relative_matrix = np.dot(np.linalg.inv(robot_trans_matrix), obs_trans_matrix)
                position = torch.tensor([relative_matrix[0:3, 3]]).to(self.device)
            phi_p, d_phi_p = self.robot_bp_sdf.build_basis_function_from_points(position, True)
       
        mesh_dict = model[0]
        weights = mesh_dict['weights'].to(self.device)
        d_s = torch.matmul(phi_p,weights)
        d_s_1 = torch.matmul(d_phi_p[:,:,0], weights)
        d_s_2 = torch.matmul(d_phi_p[:,:,1], weights)
        d_s_3 = torch.matmul(d_phi_p[:,:,2], weights)
        d_grad = torch.hstack((d_s_1, d_s_2, d_s_3))
        d_grad_abs = torch.abs(d_grad)
        d_grad_max = torch.max(d_grad_abs)
        d_grad /= d_grad_max

        return d_s, d_grad

    def test_display(self):
        self.obs_bp_sdf.create_surface_mesh(self.cylinder_model, 100, vis=True)
        self.obs_bp_sdf.create_surface_mesh(self.rect_model, 100, vis=True)
        self.obs_bp_sdf.create_surface_mesh(self.triangle_model, 100, vis=True)
        self.obs_bp_sdf.create_surface_mesh(self.pentagon_model, 100, vis=True)
        self.robot_bp_sdf.create_surface_mesh(self.robot_model, 100, vis=True)

if __name__ == '__main__':
    dc = DistanceCheck()

    robot = {'position': [0,0,0],
             'orientation': [0,0,0,1]}
    obs = {'name': 'rect',
           'position': [1,0,0],
           'orientation': [0,0,0,1]}
    
    for i in range(5):
        start_time = time.time()
        # if i % 2 == 0:
        #     obs['name'] = 'rect'
        # if i % 3 == 0:
        #     obs['name'] = 'triangle'
        info = dc.check(obs, robot)
        print(info[0], info[1])
        info = dc.check(obs, robot, reverse=True)
        print(info)
        print(time.time()-start_time)
    # plot the result

    

    

    


   


        
        


