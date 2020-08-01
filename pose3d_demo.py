import cv2
from sys import argv
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
from src.system.interface import AnnotatorInterface
from src.utils.drawer import Drawer

"""
Read the movie located at moviePath, perform the 2d pose annotation and display
Run from terminal : python pose2d_demo.py [movie_file_path] [max_persons_detected]
with all parameters optional.
Keep holding the backspace key to speed the video 30x
"""


class Skeleton:
    @staticmethod
    def parents():
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    @staticmethod
    def joints_right():
        return [1, 2, 3, 9, 10]


# noinspection PyUnresolvedReferences
def render_frame(frame, poses_3d):
    # 人数
    num_persons = len(poses_3d)
    print("num_persons: {}".format(num_persons))
    fig = plt.figure(figsize=(6 * (1 + num_persons), 6), dpi=100)
    canvas = FigureCanvasAgg(fig)
    # 输入图像
    ax_in = fig.add_subplot(1, 1 + num_persons, 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')
    ax_in.imshow(frame, aspect='equal')

    # 3D
    ax_3d_list = []
    # plot 3D axes
    for i in range(num_persons):
        ax_3d = fig.add_subplot(1, 1 + num_persons, i + 2, projection='3d')
        ax_3d.view_init(elev=15, azim=70)
        # set 长度范围
        radius = 2
        ax_3d.set_xlim3d([-radius / 2, radius / 2])
        ax_3d.set_zlim3d([0, radius])
        ax_3d.set_ylim3d([-radius / 2, radius / 2])
        ax_3d.set_aspect('equal')
        ax_3d.set_title("Reconstruction-{}".format(i + 1))
        # 坐标轴刻度
        ax_3d.set_xticklabels([])
        ax_3d.set_yticklabels([])
        ax_3d.set_zticklabels([])
        ax_3d.dist = 7.5
        ax_3d_list.append(ax_3d)

    parents = Skeleton.parents()
    joints_3d = [pose_3d.get_joints() for pose_3d in poses_3d]
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        col = 'red' if j in Skeleton.joints_right() else 'black'
        # 画图3D

        for pi, joints in enumerate(joints_3d):
            if len(joints) <= j:
                continue
            ax_3d_list[pi].plot([joints[j, 0], joints[j_parent, 0]],
                                [joints[j, 1], joints[j_parent, 1]],
                                [joints[j, 2], joints[j_parent, 2]], zdir='z', c=col)

    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    return image


def start(movie_path, max_person_num):
    annotator = AnnotatorInterface.build(max_persons=max_person_num)

    cap = cv2.VideoCapture(movie_path)
    # 原有帧率
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    pause = int(1000 / original_fps)

    while True:
        tmp_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        persons = annotator.update(frame)
        elapsed_time = time.time() - tmp_time

        fps = int(1 / elapsed_time)
        poses_2d = [p['pose_2d'] for p in persons]
        poses_3d = [p['pose_3d'] for p in persons]
        # if len(poses_2d) > 0:
        #     joints = np.zeros_like(poses_2d[0].get_joints())
        #     joints[:, 0] = (poses_2d[0].get_joints()[:, 0] * frame.shape[1])
        #     joints[:, 1] = (poses_2d[0].get_joints()[:, 1] * frame.shape[0])
        #     print("joints.shape: ", joints.shape)
        #     print('joints: ', joints)
        ids = [p['id'] for p in persons]
        frame = Drawer.draw_scene(frame, poses_2d, ids, fps, cap.get(cv2.CAP_PROP_POS_FRAMES))
        # frame = render_frame(frame, poses_3d)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) == ord(' '):
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(curr_frame + 30))

    annotator.terminate()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    print("start frontend")

    default_media = 0
    max_persons = 2

    if len(argv) == 3:
        default_media = 0 if argv[1] == "webcam" else argv[1]
        start(default_media, int(argv[2]))
    elif len(argv) == 2:
        default_media = 0 if argv[1] == "webcam" else argv[1]
        start(default_media, max_persons)
    else:
        start(default_media, max_persons)
