import cv2
from threading import Thread
from queue import Queue
import time
import numpy as np
from skimage.color import hsv2rgb
from hamiltonian import DiscreteSpace
from time_evolve import Propagator


def hsv_rep_color(probs, phis):
    """
    Map probability densities and phases onto color using HSV color representation.
    """
    h = (phis + np.pi) / 2 / np.pi
    s = np.ones_like(phis)
    v = probs

    hsv_image = np.stack([h, s, v], axis=-1)

    return hsv2rgb(hsv_image)


def render_frames(frame_queue, times, batch_size,
                  prop: Propagator,
                  space_vid: DiscreteSpace = None,
                  rescaling_factor=1.,
                  mask_potential=True,
                  fps=None):
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

    if fps is None:
        write_interval = 1
    else:
        observed_fps = len(times)/(times[-1] - times[0])
        write_interval = int(observed_fps//fps)

    # Prepare the discrete space used for video rendering
    if space_vid is None:
        space_vid = prop.hamiltonian.space
    video_size = tuple(space_vid.grid)
    print(video_size)

    # Function that is used to identify regions to be masked
    mask = prop.hamiltonian.potential(*space_vid.grid_points) if mask_potential else None

    # Loop over batches of times and render frames
    num_batches = len(times) // batch_size
    times_split = np.array_split(times, num_batches)
    norm_factor = None
    for i, batch in enumerate(times_split):
        print(f"\rGenerating time-dependent wave function. Batch {i + 1} of {len(times_split)}", end='')
        psit = prop.evolve(batch)

        # prepare the probabilities and phases
        probt = (p:=np.abs(psit)**2)/p.max(axis=(1,2), keepdims=True)
        norm_factor = probt.max()*rescaling_factor if norm_factor is None else norm_factor
        #probt /= norm_factor
        if (p_clip:=np.max(probt))>1:
            print(f"Warning: clipping detect with value {p_clip} > 1. Consider rescaling.")
        phit = np.angle(psit)

        # Use generator to feed frames into frame queue
        frames = gen_colorized_frames(probt, phit, mask_grid=mask, size=video_size, write_interval=write_interval)
        for j, f in enumerate(frames):
            frame_queue.put(f)

            # alert the user if the frame has an invalid configuration
            if f.min()<0 or f.max()>255 or \
                    f.shape!=(*video_size, 3) or \
                    f.dtype != np.uint8:
                raise Exception("Invalid frame!")


def gen_colorized_frames(probs, phis, mask_grid=None, size=None, write_interval=None):
    for i in range(0,len(probs), write_interval):
        frame = hsv_rep_color(probs[i], phis[i])
        frame = (255 * frame).astype(np.uint8)
        frame = frame[..., ::-1]  # bgr channel ordering for opencv

        # optionally resize
        if size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)

        # optionally mask the mask_func
        if mask_grid is not None:
            frame[mask_grid > 0] = np.array([128, 128, 128], dtype=np.uint8)

        yield frame


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

