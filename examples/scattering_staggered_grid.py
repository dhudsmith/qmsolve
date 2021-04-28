from functools import partial
from queue import Queue

import numpy as np
import cv2

from hamiltonian import DiscreteSpace, SingleParticle, Solver, Propagator
from video_utils import VideoWriterStream, render_frames
from potentials import multiple_hard_disks
from states import coherent_state_2d


def main():
    # ------------------------
    # Simulation Setup
    # ------------------------

    # potential
    num_cols = 3
    dot_radius = 0.20
    step = 3*dot_radius
    ys = np.arange(-7,7, step)
    #shift = step/np.sqrt(2)
    centers = [(3+i*step, y+i%2*step/2) for i in range(num_cols) for y in ys]
    rs = [dot_radius] * (len(centers))
    potential = partial(multiple_hard_disks, rs=rs, centers=centers)

    # initial state
    lam = dot_radius/2.8 #2.8*dot_radius from @quant_phys
    p = (1/lam, 0)
    xy0 = (-2, 0)
    w = (0.5, 0.5)
    init_state = partial(coherent_state_2d, p=p, xy0=xy0, w=w)

    # system and solver
    dim = 2  # spacial dimension
    support = (-6, 6)  # support region of mask_func
    grid = 325  # number of grid points along one dimension. Assumed square.
    dtype = np.float64  # datatype used for internal processing
    num_states = 2500  # how many eigenstates to consider for time evolution
    method = 'eigsh'  # eigensolver method. One of 'eigsh' or 'lobpcg'

    # video arguments
    name = 'scattering_staggered_grid'
    rescaling_factor = 1
    fps = 30
    times = np.concatenate([np.zeros(1 * fps), np.linspace(0, 1.5, 12 * fps)])
    batch_size = fps
    grid_video = 1080
    video_size = (grid_video, grid_video)
    fourcc_str = 'mp4v'
    extension = 'mp4'
    video_file = f"../assets/{name}.{extension}"

    # ------------------------
    # Simulation objects
    # ------------------------

    space = DiscreteSpace(dim, support, grid, dtype)
    space_vid = DiscreteSpace(dim, support, grid_video, dtype)
    ham = SingleParticle(space, potential)
    solver = Solver(method=method)
    prop = Propagator(ham, init_state, solver, num_states)

    # ------------------------
    # Run simulation and create outputs
    # ------------------------

    # initiate the frame writing pipeline
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out = cv2.VideoWriter(filename=video_file,
                          fourcc=fourcc,
                          fps=fps,
                          frameSize=video_size,
                          isColor=True)
    write_queue = Queue()
    vws = VideoWriterStream(out, write_queue)
    thread = vws.start()

    # render the frames to the write_queue
    render_frames(write_queue, times, batch_size, prop, space_vid, rescaling_factor)

    # shutdown the thread
    vws.stop()
    thread.join()


if __name__ == '__main__':
    main()
