import cv2
import numpy as np

from .keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS, a


class Pose:
    num_kpts = 18
    kpt_names = ["0:  nose_id",
                 "1:  neck_id",

                 "2:  r_shoulder_id",
                 "3:  r_elbow_id",
                 "4:  r_wrist_id",
                 "8:  r_hip_id",
                 "9:  r_knee_id",
                 "10: r_ankle_id",
                 "14: r_eye_id",
                 "16: r_ear_id",

                 "5:  l_shoulder_id",
                 "6:  l_elbow_id",
                 "7:  l_wrist_id",
                 "11: l_hip_id",
                 "12: l_knee_id",
                 "13: l_ankle_id",
                 "15: l_eye_id",
                 "17: l_ear_id"]
    print("0:  nose_id",
          "\n1:  neck_id",

          "\n2:  r_shoulder_id",
          "\n3:  r_elbow_id",
          "\n4:  r_wrist_id",
          "\n8:  r_hip_id",
          "\n9:  r_knee_id",
          "\n10: r_ankle_id",
          "\n14: r_eye_id",
          "\n16: r_ear_id",

          "\n5:  l_shoulder_id",
          "\n6:  l_elbow_id",
          "\n7:  l_wrist_id",
          "\n11: l_hip_id",
          "\n12: l_knee_id",
          "\n13: l_ankle_id",
          "\n15: l_eye_id",
          "\n17: l_ear_id")
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    color1 = [0, 0, 0]

    color2 = [255, 255, 255]

    color3 = [0, 1, 255]

    color4 = [0, 255, 0]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None

    @staticmethod
    def get_bbox(keypoints):
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)

        return bbox

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1
            # print("Pose.last_id, ",Pose.last_id)

    def draw(self, img):

        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            right = [2, 3, 4, 8, 9, 10, 14, 16]
            left = [5, 6, 7, 11, 12, 13, 15, 17]
            neck=[1]
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                a = [kpt_a_id, (x_a, y_a)]
                print(a)
                "0:  nose_id"
                "1:  neck_id"

                "2:  r_shoulder_id"
                "3:  r_elbow_id"
                "4:  r_wrist_id"
                "8:  r_hip_id"
                "9:  r_knee_id"
                "10: r_ankle_id"
                "14: r_eye_id"
                "16: r_ear_id"

                "5:  l_shoulder_id"
                "6:  l_elbow_id"
                "7:  l_wrist_id"
                "11: l_hip_id"
                "12: l_knee_id"
                "13: l_ankle_id"
                "15: l_eye_id"
                "17: l_ear_id"

                # if kpt_a_id == 2:
                #    # a=[kpt_a_id,(x_a,y_a)]
                #    print("r_shoulder_id", kpt_a_id, ":", x_a, y_a)
                #    cv2.putText(img, 'r_shoulder', (x_a + 9, y_a), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #    # print(a)
                #
                # if kpt_a_id == 3:
                #    # b= [kpt_a_id, (x_a, y_a)]
                #    print("r_elbow_id", kpt_a_id, ":", x_a, y_a)
                #    cv2.putText(img, 'r_elbow', (x_a + 9, y_a), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #
                # if kpt_a_id == 4:
                #    # c = [kpt_a_id, (x_a, y_a)]
                #    print("r_wrist_id", kpt_a_id, ":", x_a, y_a)
                #    cv2.putText(img, 'r_wrist', (x_a + 9, y_a), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #
                # if kpt_a_id == 8:
                #    # d= [kpt_a_id, (x_a, y_a)]
                #    print("r_hip_id", kpt_a_id, ":", x_a, y_a)
                #    cv2.putText(img, 'r_hip', (x_a + 9, y_a), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #
                # if kpt_a_id == 9:
                #    # e= [kpt_a_id, (x_a, y_a)]
                #    print("r_knee_id", kpt_a_id, ":", x_a, y_a)
                #    cv2.putText(img, 'r_knee', (x_a + 9, y_a), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #
                # if kpt_a_id == 10:
                #    # f= [kpt_a_id, (x_a, y_a)]
                #    print("r_ankle_id", kpt_a_id, ":", x_a, y_a)
                #    cv2.putText(img, 'r_ankle', (x_a + 9, y_a), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #
                # if kpt_a_id == 14:
                #    # g= [kpt_a_id, (x_a, y_a)]
                #    print("r_eye_id", kpt_a_id, ":", x_a, y_a)
                #    cv2.putText(img, 'r_eye', (x_a + 9, y_a), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #
                # if kpt_a_id == 16:
                #    # h= [kpt_a_id, (x_a, y_a)]
                #    print("r_ear_id", kpt_a_id, ":", x_a, y_a)
                #    cv2.putText(img, 'r_ear', (x_a + 9, y_a), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

                if kpt_a_id in right:
                    cv2.circle(img, (int(x_a), int(y_a)), 2, Pose.color4, -1)#3
                if kpt_a_id in neck:
                    cv2.circle(img, (int(x_a), int(y_a)), 2, (255, 200, 1), -1)
                    xy = "%d,%d" % (x_a, y_a)
                    # cv2.putText(img, xy, (x_a + 9, y_a), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 200, 1), thickness=1)
                    print("neck_id", kpt_a_id, ":", x_a, y_a)
                    # cv2.putText(img, 'neck', (x_a + 9, y_a-19), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,200,1), 1)

                else:
                    cv2.circle(img, (int(x_a), int(y_a)), 2, Pose.color4, -1)#no

            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]

                # if kpt_b_id == 2:
                #    # a=[kpt_a_id,(x_a,y_a)]
                #    print("r_shoulder_id", kpt_b_id, ":", x_b, y_b)
                #    cv2.putText(img, 'r_shoulder', (x_b + 9, y_b), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #    # print(a)
                #
                # if kpt_b_id == 3:
                #    # b= [kpt_a_id, (x_a, y_a)]
                #    print("r_elbow_id", kpt_b_id, ":", x_b, y_b)
                #    cv2.putText(img, 'r_elbow', (x_b + 9, y_b), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #
                # if kpt_b_id == 4:
                #    # c = [kpt_b_id, (x_a, y_a)]
                #    print("r_wrist_id", kpt_b_id, ":", x_b, y_b)
                #    cv2.putText(img, 'r_wrist', (x_b + 9, y_b), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #
                # if kpt_b_id == 8:
                #    # d= [kpt_a_id, (x_a, y_a)]
                #    print("r_hip_id", kpt_b_id, ":", x_b, y_b)
                #    cv2.putText(img, 'r_hip', (x_b + 9, y_b), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #
                # if kpt_b_id == 9:
                #    # e= [kpt_a_id, (x_a, y_a)]
                #    print("r_knee_id", kpt_b_id, ":", x_b, y_b)
                #    cv2.putText(img, 'r_knee', (x_b + 9, y_b), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #
                # if kpt_b_id == 10:
                #    # f= [kpt_a_id, (x_a, y_a)]
                #    print("r_ankle_id", kpt_b_id, ":", x_b, y_b)
                #    cv2.putText(img, 'r_ankle', (x_b + 9, y_b), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #
                # if kpt_b_id == 14:
                #    # g= [kpt_a_id, (x_a, y_a)]
                #    print("r_eye_id", kpt_b_id, ":", x_b, y_b)
                #    cv2.putText(img, 'r_eye', (x_b + 9, y_b), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                #
                # if kpt_b_id == 16:
                #    # h= [kpt_a_id, (x_a, y_a)]
                #    print("r_ear_id", kpt_b_id, ":", x_b, y_b)
                #    cv2.putText(img, 'r_ear', (x_b + 9, y_b), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

                if kpt_b_id in right:
                    cv2.circle(img, (int(x_b), int(y_b)), 4, Pose.color4, -1)#3
                if kpt_b_id in neck:
                    cv2.circle(img, (int(x_b), int(y_b)), 4, (255, 200, 1), -1)
                    # print("r_ear_id", kpt_a_id, ":", x_a, y_a)
                    # cv2.putText(img, 'neck', (x_b + 9, y_b-19), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 200, 1), 1)
                    xy = "%d,%d" % (x_b, y_b)
                    # cv2.putText(img, xy, (x_b + 9, y_b), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 200, 1), 1)
                else:
                    cv2.circle(img, (int(x_b), int(y_b)), 4, Pose.color4, -1)#no
                # xy = "%d,%d" % (x_b, y_b)
                # cv2.putText(img, xy, (x_b + 5, y_b), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
                # cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color1, -1)

            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                if kpt_a_id in right or kpt_b_id in right:
                    cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (0, 0, 255), 2)
                else:
                    cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 2)


def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt
