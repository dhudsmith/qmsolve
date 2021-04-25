from functools import partial
import numpy as np
import cv2
from queue import Queue

from hamiltonian import DiscreteSpace, SingleParticle, Solver, Propagator
from video_utils import VideoWriterStream, render_frames
from potentials import multiple_hard_disks
from states import coherent_state_2d


def main():
    # ------------------------
    # Simulation Setup
    # ------------------------

    # potential
    rs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    centers = [(0, -3), (0, -2), (0, -1.), (0, 0), (0, 1.), (0, 2), (0, 3)]
    potential = partial(multiple_hard_disks, rs=rs, centers=centers)

    # initial state
    p = (6, 0)
    xy0 = (-4, 0)
    init_state = partial(coherent_state_2d, p=p, xy0=xy0)

    # system and solver
    dim = 2  # spacial dimension
    support = (-5, 5)  # support region of mask_func
    grid = 300  # number of grid points along one dimension. Assumed square.
    dtype = np.float64  # datatype used for internal processing
    num_states = 700  # how many eigenstates to consider for time evolution
    method = 'eigsh'  # eigensolver method. One of 'eigsh' or 'lobpcg'

    # video arguments
    name = 'scattering_circular_barrier'
    rescaling_factor = 1
    fps = 30
    times = np.concatenate([np.zeros(1 * fps), np.linspace(0, 2, 7 * fps)])
    batch_size = 360
    grid_video = 720
    video_size = (grid_video, grid_video)
    fourcc_str = 'FFV1'
    extension = 'avi'
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
