import torch
import warp as wp
import numpy as np
import matplotlib.pyplot as plt

@wp.kernel
def add_pixel_dropout_and_randu_kernel(
    # inputs
    depths: wp.array(dtype=float, ndim=3),
    rand_dropout: wp.array(dtype=float, ndim=3),
    rand_u: wp.array(dtype=float, ndim=3),
    rand_u_values: wp.array(dtype=float, ndim=3),
    p_dropout: float,
    p_randu: float,
    d_min: float,
    d_max: float,
    kernel_size: int,
    seed: int):

    batch_index, pixel_row, pixel_column = wp.tid()

    # Perform dropout
    if rand_dropout[batch_index, pixel_row, pixel_column] <= p_dropout:
        depths[batch_index, pixel_row, pixel_column] = 0.

    # Insert random uniform value
    if rand_u[batch_index, pixel_row, pixel_column] <= p_randu:
        rand_depth =\
            rand_u_values[batch_index, pixel_row, pixel_column] * (d_max - d_min) + d_min
        depths[batch_index, pixel_row, pixel_column] = rand_depth

        for i in range(kernel_size):
            for j in range(kernel_size):
                state =\
                    wp.rand_init(seed, batch_index + pixel_row + pixel_column + i + j)
                if wp.randf(state) < 0.25:
                    depths[batch_index, pixel_row + i, pixel_column + j] = rand_depth

@wp.kernel
def add_sticks_kernel(
    # inputs
    depths: wp.array(dtype=float, ndim=3),
    rand_sticks: wp.array(dtype=float, ndim=3),
    rand_sticks_depths: wp.array(dtype=float, ndim=3),
    p_stick: float,
    max_stick_len: float,
    max_stick_width: float,
    height: int,
    width: int,
    d_min: float,
    d_max: float,
    seed: int):

    batch_index, pixel_row, pixel_column = wp.tid()

    stick_width = float(0.)
    stick_len = float(0.)
    stick_rot = float(0.)
        
    rand_depth =\
        rand_sticks_depths[batch_index, pixel_row, pixel_column] * (d_max - d_min) + d_min
    
    if rand_sticks[batch_index, pixel_row, pixel_column] <= p_stick:
        for i in range(3):
            state =\
                wp.rand_init(seed, batch_index + pixel_row + pixel_column + i)

            if i == 0:
                stick_width = wp.randf(state) * max_stick_width
            if i == 1:
                stick_len = wp.randf(state) * max_stick_len + 1.
            if i == 2:
                stick_rot = wp.randf(state) * (3.14 * 2.)

        # Set origin of stick
        #hor_coord = float(pixel_column)
        #vert_coord = float(pixel_row)

        #slope = wp.sin(stick_rot) / wp.cos(stick_rot)

        for i in range(int(wp.rint(stick_len))):
            hor_coord = float(pixel_column + i)
            vert_coord = wp.floor(float(i) * wp.sin(stick_rot)) + float(pixel_row)

            # Bound
            if hor_coord > float(width):
                hor_coord = float(width)
            if hor_coord < 0.:
                hor_coord = 0.
            if vert_coord > float(height):
                vert_coord = float(height)
            if vert_coord < 0.:
                vert_coord = 0.

            depths[batch_index, int(vert_coord), int(hor_coord)] = rand_depth
            #depths[batch_index, int(vert_coord + 1.), int(hor_coord + 1.)] = rand_depth

            for j in range(1, int(max_stick_width)):
                if stick_rot > (3.14 / 4.) and stick_rot < (3. * 3.14 / 4.):
                    depths[batch_index, int(vert_coord) + j, int(hor_coord)] = rand_depth
                    #depths[batch_index, int(vert_coord) + 2, int(hor_coord) + 1] = rand_depth
                elif stick_rot > (5. * 3.14 / 4.) and stick_rot < (7. * 3.14 / 4.):
                    depths[batch_index, int(vert_coord) + j, int(hor_coord)] = rand_depth
                    #depths[batch_index, int(vert_coord) + 2, int(hor_coord) + 1] = rand_depth
                else:
                    depths[batch_index, int(vert_coord), int(hor_coord) + j] = rand_depth
                    #depths[batch_index, int(vert_coord)+1, int(hor_coord) + 2] = rand_depth

@wp.kernel
def add_correlated_noise_kernel(
    # inputs
    depths: wp.array(dtype=float, ndim=3),
    rand_sigma_s_x: wp.array(dtype=float, ndim=3),
    rand_sigma_s_y: wp.array(dtype=float, ndim=3),
    rand_sigma_d: wp.array(dtype=float, ndim=3),
    height: int,
    width: int,
    d_min: float,
    d_max: float,
    # outputs
    noisy_depths: wp.array(dtype=float, ndim=3)):

    batch_index, pixel_row, pixel_column = wp.tid()

    # Draw float pixel coordinates
    nx = rand_sigma_s_x[batch_index, pixel_row, pixel_column]
    ny = rand_sigma_s_y[batch_index, pixel_row, pixel_column]

    u = nx + float(pixel_column)
    v = ny + float(pixel_row)

    u0 = int(u)
    v0 = int(v)
    u1 = u0 + 1
    v1 = v0 + 1

    fu = u - float(u0)
    fv = v - float(v0)

    # Ensure bounds
    u0 = wp.max(0, wp.min(u0, width - 1))
    u1 = wp.max(0, wp.min(u1, width - 1))
    v0 = wp.max(0, wp.min(v0, height - 1))
    v1 = wp.max(0, wp.min(v1, height - 1))

    # Linear interp weights
    w_00 = (1. - fu) * (1. - fv)
    w_01 = (1. - fu) * fv
    w_10 = fu * (1. - fv)
    w_11 = fu * fv

    # Interpolated depth
    noisy_depths[batch_index, pixel_row, pixel_column] =\
        depths[batch_index, v0, u0] * w_00 +\
        depths[batch_index, v0, u1] * w_01 +\
        depths[batch_index, v1, u0] * w_10 +\
        depths[batch_index, v1, u1] * w_11
    

    #noisy_depths[batch_index, pixel_row, pixel_column]

    baseline = float(35130.)
    ref = float(8.)

    denominator = baseline / noisy_depths[batch_index, pixel_row, pixel_column] +\
                  rand_sigma_d[batch_index, pixel_row, pixel_column] + 0.5
    
    noisy_depths[batch_index, pixel_row, pixel_column] =\
        baseline / (wp.rint(denominator / ref) * ref)
        #float(int(baseline) / int(wp.rint(denominator / ref)) * int(ref))
    #noisy_depths[batch_index, pixel_row, pixel_column] =\
    #    baseline / denominator

@wp.kernel
def add_normal_noise_kernel(
    # inputs
    depths: wp.array(dtype=float, ndim=3),
    rand_sigma_theta: wp.array(dtype=float, ndim=3),
    cam_matrix: wp.mat44f,
    height: int,
    width: int,
    d_min: float,
    d_max: float):

    batch_index, pixel_row, pixel_column = wp.tid()
    
    if pixel_row == (height - 1):
        return
    if pixel_column == (width - 1):
        return

    # Get 3D point at current pixel
    uv = wp.vec4f(float(pixel_column), float(pixel_row), 1., 1.)
    xyzw = wp.inverse(cam_matrix) * uv

    x_hat = xyzw[0] / xyzw[3]
    y_hat = xyzw[1] / xyzw[3]
    z_hat = xyzw[2] / xyzw[3]

    d = depths[batch_index, pixel_row, pixel_column]
    point_3d = wp.vec3f(d * x_hat, d * y_hat, d * z_hat)

    # Get 3D point of one pixel to the right
    uv = wp.vec4f(float(pixel_column + 1), float(pixel_row), 1., 1.)
    xyzw = wp.inverse(cam_matrix) * uv

    x_hat = xyzw[0] / xyzw[3]
    y_hat = xyzw[1] / xyzw[3]
    z_hat = xyzw[2] / xyzw[3]

    d = depths[batch_index, pixel_row, pixel_column + 1]
    point_3d_01 = wp.vec3f(d * x_hat, d * y_hat, d * z_hat)
    
    # Get 3D point of one pixel up
    uv = wp.vec4f(float(pixel_column), float(pixel_row + 1), 1., 1.)
    xyzw = wp.inverse(cam_matrix) * uv

    x_hat = xyzw[0] / xyzw[3]
    y_hat = xyzw[1] / xyzw[3]
    z_hat = xyzw[2] / xyzw[3]

    d = depths[batch_index, pixel_row + 1, pixel_column]
    point_3d_10 = wp.vec3f(d * x_hat, d * y_hat, d * z_hat)

    # Find normal of three points

    x_axis = wp.normalize(point_3d_01 - point_3d)
    y_axis = wp.normalize(point_3d_10 - point_3d)

    normal = wp.cross(x_axis, y_axis)
    
    #if pixel_row == 90 and pixel_column == 100:
    #    print(point_3d)

    # Now perturb 3D coordinate by noise amount along normal
    point_3d =\
        point_3d + 1000. * rand_sigma_theta[batch_index, pixel_row, pixel_column] * normal
    #if pixel_row == 90 and pixel_column == 100:
    #    print(point_3d)
    #    print('----')

    # Take the z value as depth in meters
    depths[batch_index, pixel_row, pixel_column] = -point_3d[2] / 1000.

class DepthAug():
    def __init__(self, device):
        self.device = device

        # Generate RNG state
        #seed = torch.rand(1).int().item()
        #self.state = wp.rand_init(seed)
        self.seed = 42
        self.kernel_size = 2

    def add_pixel_dropout_and_randu(self, depths, p_dropout, p_randu, d_min, d_max):
        
        batch_size = depths.shape[0]
        height = depths.shape[1]
        width = depths.shape[2]

        rand_dropout = torch.rand(batch_size, height, width, device=self.device)
        rand_u = torch.rand(batch_size, height, width, device=self.device)
        rand_u_values = torch.rand(batch_size, height, width, device=self.device)
        
        wp.launch(kernel=add_pixel_dropout_and_randu_kernel,
                  dim=[batch_size, height, width],
                  inputs=[
                      wp.torch.from_torch(depths),
                      wp.torch.from_torch(rand_dropout),
                      wp.torch.from_torch(rand_u),
                      wp.torch.from_torch(rand_u_values),
                      p_dropout,
                      p_randu,
                      d_min,
                      d_max,
                      self.kernel_size,
                      self.seed],
                  device=self.device)

    def add_sticks(self, depths, p_stick, max_stick_len, max_stick_width, d_min, d_max):
        batch_size = depths.shape[0]
        height = depths.shape[1]
        width = depths.shape[2]

        rand_stick = torch.rand(batch_size, height, width, device=self.device)
        rand_stick_depths = torch.rand(batch_size, height, width, device=self.device)

        wp.launch(kernel=add_sticks_kernel,
                  dim=[batch_size, height, width],
                  inputs=[
                      wp.torch.from_torch(depths),
                      wp.torch.from_torch(rand_stick),
                      wp.torch.from_torch(rand_stick_depths),
                      p_stick,
                      max_stick_len,
                      max_stick_width,
                      height,
                      width,
                      d_min,
                      d_max,
                      self.seed],
                  device=self.device)

    # NOTE: taken from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6907054
    def add_correlated_noise(self, depths, noisy_depths, sigma_s, sigma_d, d_min, d_max):
        batch_size = depths.shape[0]
        height = depths.shape[1]
        width = depths.shape[2]
        
        rand_sigma_s_x = sigma_s * torch.randn(batch_size, height, width, device=self.device)
        rand_sigma_s_y = sigma_s * torch.randn(batch_size, height, width, device=self.device)
        rand_sigma_d = sigma_d * torch.randn(batch_size, height, width, device=self.device)

        wp.launch(kernel=add_correlated_noise_kernel,
                  dim=[batch_size, height, width],
                  inputs=[
                      wp.torch.from_torch(depths),
                      wp.torch.from_torch(rand_sigma_s_x),
                      wp.torch.from_torch(rand_sigma_s_y),
                      wp.torch.from_torch(rand_sigma_d),
                      height,
                      width,
                      d_min,
                      d_max],
                  outputs=[
                      wp.torch.from_torch(noisy_depths)],
                  device=self.device)

    def add_normal_noise(self, depths, sigma_theta, cam_matrix, d_min, d_max):
        batch_size = depths.shape[0]
        height = depths.shape[1]
        width = depths.shape[2]
        
        rand_sigma_theta =\
            sigma_theta * torch.randn(batch_size, height, width, device=self.device)

        wp.launch(kernel=add_normal_noise_kernel,
                  dim=[batch_size, height, width],
                  inputs=[
                      wp.torch.from_torch(depths),
                      wp.torch.from_torch(rand_sigma_theta),
                      cam_matrix,
                      height,
                      width,
                      d_min,
                      d_max],
                  device=self.device)
        
# Example of how to apply depth randomizations
if __name__ == "__main__":
    wp.init()
    device = 'cuda'
    depth_aug = DepthAug(device)

    # Dropout and random noise blob parameters
    p_dropout = 0.0125 / 4
    p_randu = 0.0125 / 4
    d_max = 1.5
    d_min = 0.5

    # Random stick parameters
    p_stick = 0.001 / 4
    max_stick_len = 18. # in pixels
    max_stick_width = 3. # in pixels

    # Correlated noise parameters
    sigma_s = 1./2
    sigma_d = 1./6

    # Normal noise parameters
    cam_matrix = wp.mat44f()
    cam_matrix[0,0] = 2.2460368
    cam_matrix[1, 1] = 2.9947157
    cam_matrix[2, 3] = -1.
    cam_matrix[3, 2] = 1.e-3

    sigma_theta = 0.01
    
    # Read in depth from file
    depth = np.load('depth_squirrel.npy')
    depth = np.load('test_depth_env.npy')[0]

    depths_raw = torch.tensor([depth], device=device)

    WIDTH = depths_raw.shape[2]
    HEIGHT = depths_raw.shape[1]

    # Create visualizer
    fig = plt.figure()
    x = np.linspace(0,50.,num=WIDTH)
    y = np.linspace(0,50.,num=HEIGHT)
    X,Y = np.meshgrid(x,y)
    ax = fig.add_subplot(1,1,1)
    rendered_img = ax.imshow(X, vmin=0,vmax=1.5, cmap='Greys')
    fig.canvas.draw()
    plt.title("Input")
    plt.show(block=False)

    input('Press ENTER to continue')

    for i in range(1000):
        # This is the raw clean depths
        depths = torch.clone(depths_raw)
        
        # This adds correlated noise to depths
        depth_aug.add_correlated_noise(
            depths_raw, depths, sigma_s, sigma_d, d_min, d_max)

        # Add normal noise
        depth_aug.add_normal_noise(depths, sigma_theta, cam_matrix, d_min, d_max)

        # This add pixel dropout and random uniform depth values
        depth_aug.add_pixel_dropout_and_randu(depths, p_dropout, p_randu, d_min, d_min)
        
        # This adds random sticks to depth
        depth_aug.add_sticks(depths, p_stick, max_stick_len, max_stick_width, d_min, d_max)
        
        # Render 
        depth = depths[0].detach().cpu().numpy()
        rendered_img.set_data(depth)
        fig.canvas.draw()
        fig.canvas.flush_events()
