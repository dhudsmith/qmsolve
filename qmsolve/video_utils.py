import cv2
from threading import Thread
from queue import Queue
import time
import numpy as np
from skimage.color import hsv2rgb
from hamiltonian import DiscreteSpace
from time_evolve import Propagator
from typing import Generator


def hsv_rep_color(probs, phis):
    """
    Map probability densities and phases onto color using HSV color representation.
    """
    h = (phis + np.pi) / 2 / np.pi
    s = np.ones_like(phis)
    v = probs

    hsv_image = np.stack([h, s, v], axis=-1)

    return hsv2rgb(hsv_image)


def render_frames(frame_queue,
                  psit_gen: Generator[np.ndarray, None, None],
                  duration,
                  fps,
                  space_vid,
                  mask_grid=None,
                  progress_interval=None):
    """
    Simulate dynamics and render frames into the provided frame queue.
    :param frame_queue: A queue to hold simulation results.
    :param times: The times for which the wave function is computed.
    :param batch_size: The number of frames simultaneously processed. Larger is
        faster but takes more memory.
    :param prop: The propagator object used to solve the time-dependent
        Schrodinger equation
    :param space_vid: the space object used to control the video resolution
    :param rescaling_factor: Scales the probabilities to avoid clipping.
    :param mask_potential: Whether or not to use the potential to mask regions of the
        rendered frames. Useful for scattering off of hard objects.
    :return:
    """

    if not progress_interval:
        progress_interval = int(fps//2)

    # Prepare the discrete space used for video rendering
    video_size = tuple(space_vid.grid)

    # Function that is used to identify regions to be masked
    #mask = prop.hamiltonian.potential(*space_vid.grid_points) if mask_potential else None

    # Loop over batches of times and render frames
    i = 0
    t = 0.
    while t<=duration:
        # prepare the probabilities and phases
        psi = next(psit_gen)
        frame = prep_frame(psi, mask_grid=mask_grid, size=video_size)
        frame_queue.put(frame)

        t += 1 / fps
        i += 1

        if i % progress_interval == 0:
            print(f"\rProcessed up to time {t:0.2f} of {duration}. {t/duration:0.1%} complete.", end='')


def prep_frame(psi, mask_grid=None, size=None):
    prob = (p := np.abs(psi) ** 2) / p.max()
    if (p_clip := np.max(prob)) > 1:
        print(f"Warning: clipping detect with value {p_clip} > 1. Consider rescaling.")
    phi = np.angle(psi)

    # optionally resize
    if size:
        prob = cv2.resize(prob, size, interpolation=cv2.INTER_LANCZOS4)
        phi = cv2.resize(phi, size, interpolation=cv2.INTER_LANCZOS4)

    frame = hsv_rep_color(prob, phi)
    frame = (255 * frame).astype(np.uint8)
    frame = frame[..., ::-1]  # bgr channel ordering for opencv

    # optionally mask the mask_func
    if mask_grid is not None:
        frame[mask_grid > 0] = np.array([128, 128, 128], dtype=np.uint8)

    # alert the user if the frame has an invalid configuration
    if frame.min() < 0 or frame.max() > 255 or \
            frame.shape != (*size, 3) or \
            frame.dtype != np.uint8:
        raise Exception("Invalid frame!")

    return frame


class VideoWriterStream:
    def __init__(self, writer: cv2.VideoWriter, frame_queue: Queue, name="VideoWriter"):

        self.writer = writer
        self.frame_queue = frame_queue
        self.name = name

        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return t

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.writer.release()
                return

            # otherwise, write the next frame
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.writer.write(frame)
            else:
                time.sleep(0.01)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

