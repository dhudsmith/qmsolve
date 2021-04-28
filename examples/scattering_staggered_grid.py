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
    num_dots_per_col = 10
    num_cols = 5
    rs = [0.22]*(num_dots_per_col * num_cols)
    ys = np.linspace(-8, 8, num_dots_per_col)
    step = ys[1]-ys[0]
    shift = step/np.sqrt(2)
    centers = [(0+i*shift, y+i%2*step/2) for i in range(num_cols) for y in ys]
    potential = partial(multiple_hard_disks, rs=rs, centers=centers)

    # initial state
    p = (6, 0)
    xy0 = (-4, 0)
    w = (0.5, 0.5)
    init_state = partial(coherent_state_2d, p=p, xy0=xy0, w=w)

    # system and solver
    dim = 2  # spacial dimension
    support = (-6, 6)  # support region of mask_func
    grid = 200  # number of grid points along one dimension. Assumed square.
    dtype = np.float64  # datatype used for internal processing
    num_states = 900  # how many eigenstates to consider for time evolution
    method = 'eigsh'  # eigensolver method. One of 'eigsh' or 'lobpcg'

    # video arguments
    name = 'scattering_staggered_grid'
    rescaling_factor = 1
    fps = 30
    times = np.concatenate([np.zeros(1 * fps), np.linspace(0, 2.5, 20 * fps)])
    batch_size = 2*fps
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
