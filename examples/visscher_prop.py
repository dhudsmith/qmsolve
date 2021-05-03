from functools import partial
from queue import Queue

import numpy as np
import cv2

from hamiltonian import DiscreteSpace, SingleParticle
from time_evolve import VisscherPropagator as VissProp
from video_utils import VideoWriterStream, render_frames
from potentials import multiple_hard_disks
from states import coherent_state_2d


def main():
    # ------------------------
    # Simulation Setup
    # ------------------------

    # potential
    num_dots = 4
    rs = [0.6666]*num_dots
    cys = np.linspace(-4,4, num_dots)
    centers = [(2, cy) for cy in cys]
    scales = [30]*num_dots
    potential = partial(multiple_hard_disks, rs=rs, centers=centers, scales=scales)

    # initial state
    p = (2, 0)
    xy0 = (-4, 0)
    w = (0.5, 0.5)
    init_state = partial(coherent_state_2d, p=p, xy0=xy0, w=w)

    # system and solver
    dim = 2  # spacial dimension
    support = (-6, 6)  # support region of mask_func
    grid = 50  # number of grid points along one dimension. Assumed square.
    dtype = np.float32  # datatype used for internal processing
    dt = 0.0003
    times = np.arange(0, 8, dt)

    # video arguments
    name = 'visscher_prop'
    rescaling_factor = 1
    fps = 30
    batch_size = len(times)
    grid_video = 480
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
    prop = VissProp(ham, init_state)

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
    render_frames(write_queue, times, batch_size, prop, space_vid, rescaling_factor, fps=fps)

    # shutdown the thread
    vws.stop()
    thread.join()


if __name__ == '__main__':
    main()
