import numpy as np
import dlib
import cv2

'# load the predictor for 68 face landmark shape predictor'
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
'# load face detector'
face_detector = dlib.get_frontal_face_detector()


def face_detection(src, accuracy=0):
    """

    :param accuracy: a higher value detects more faces
    :param src: image where the faces should be located
    :return: a array where all dst of the faces in bsp: faces[0] = first detected face on the image
    """
    frame_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces = face_detector(frame_gray, accuracy)

    return faces


def show_faces(src, faces, color=(0, 255, 0), thick=3):
    """

    :param src: frame in BGR format
    :param faces: dlib face object
    :param color:  color of the rectangle
    :param thick: thickness of the rectangle
    :return:
    """
    frame = src.copy()
    for face in faces:
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, thick)

    return frame


def get_landmarks(src, faces):
    """

    :param src: image with the face on it
    :param faces: dst of the face on the image
    :return: a list of all 68 landmark points
    """
    gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces_landmark_list = list()

    for face in faces:
        landmarks = predictor(gray_image, face)

        landmarks_list = list()

        for land in range(68):
            landmarks_list.append((landmarks.part(land).x, landmarks.part(land).y))

        faces_landmark_list.append(landmarks_list)

    return faces_landmark_list


def show_landmarks(src, faces_landmarks_list, color=(0, 255, 0)):
    """
        This function is only to debug the program

    :param faces_landmarks_list: list of all 68 landmarks in the face
    :param src: image where the points should be drawn
    :param color: color of the points
    :return: the given image with the landmark points drawn on it
    """
    image = src.copy()
    for face in faces_landmarks_list:
        for point in face:
            image = cv2.circle(src, point, 5, color, -1)

    return image


def create_mask(image, points):
    """

    :param image: image where the mask should be on, it is need for size of the mask
    :param points: dst points for the mask
    :return: return a sws mask which is same size as the src image
    """
    width, height, channels = image.shape

    mask = np.zeros((width, height, channels), np.uint8)

    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, (255, 255, 255))

    return mask


def get_mouth_and_eye_points(landmarks_list):
    """

    :param landmarks_list: list of the landmark points of the face
    :return: the position of the mouth and eye points in from from a np.array()
    """
    mouth_points = np.array([landmarks_list[48], landmarks_list[49], landmarks_list[50], landmarks_list[51],
                             landmarks_list[52], landmarks_list[53], landmarks_list[54], landmarks_list[55],
                             landmarks_list[56], landmarks_list[57], landmarks_list[58], landmarks_list[59],
                             landmarks_list[60]], np.int32)

    left_eye = np.array([landmarks_list[42], landmarks_list[43], landmarks_list[44], landmarks_list[45],
                         landmarks_list[46], landmarks_list[47]], np.int32)

    right_eye = np.array([landmarks_list[36], landmarks_list[37], landmarks_list[38], landmarks_list[39],
                          landmarks_list[40], landmarks_list[41]], np.int32)

    return mouth_points, left_eye, right_eye


def create_round_mask(src, points):
    """

    :param src: source image (for checking the size of the mask)
    :param points: points for the mask (the mask is a circle which contain all the points)
    :return: the mask as sw image
    """
    mask = np.zeros_like(src)
    center, radius = cv2.minEnclosingCircle(points)
    mask = cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), color=(255, 255, 255), thickness=-1)

    return mask


def get_eyes_and_mouth_mask(src, faces_landmarks_list):
    """

    :param src: source image with the faces on (frame from the camara)
    :param faces_landmarks_list: landmark list of the face
    :return: a mask of mouth and eyes in range 0..255 255 is face and mouth 0 is the background
    """
    mask = np.zeros_like(src)
    for landmarks_list in faces_landmarks_list:
        mouth_points, left_eye, right_eye = get_mouth_and_eye_points(landmarks_list)

        mask_mouth = create_mask(mask, mouth_points)
        mask_left_eye = create_round_mask(mask, left_eye)
        mask_right_eye = create_round_mask(mask, right_eye)

        mask = mask + mask_mouth + mask_left_eye + mask_right_eye

    return mask


def cut_out_mask(src, mask):
    """

    :param src: source image with the faces on (frame from the camara)
    :param mask: mask, of the item which is to crop
    :return: the cropped image (fill color from src image into the white points of the mask image)
    """
    cut_image = np.zeros_like(src)
    cut_image[np.where(mask == 255)] = src[np.where(mask == 255)]

    return cut_image


def resize_image(src, new_size):
    """

    :param src: image which is to resize
    :param new_size: new size for the image
    :return: the image with the new size
    """
    new_image = cv2.resize(src.copy(), new_size)

    return new_image


def cut_rectangle_from_points(cut_item, mask, points, show=False):
    """

    :param mask: mask of the cut item
    :param cut_item: the image of the cut item
    :param show: when True it shows the mask before continue the code
    :param points:  contours of the cut item
    :return:  a mask of the cut item and the item itself
    """

    x_point, y_point, w_point, h_point = cv2.boundingRect(points)
    item = cut_item[y_point: y_point + h_point, x_point: x_point + w_point]
    item_mask = mask[y_point: y_point + h_point, x_point: x_point + w_point]

    if show is True:
        cv2.imshow('item', item)
        cv2.imshow('item_mask', item_mask)
        cv2.waitKey(0)
        cv2.destroyWindow('item')
        cv2.destroyWindow('item_mask')

    return item, item_mask


def cut_circle_from_points(cut_item, mask, points, show=False):
    """

    :param cut_item: item which is cut
    :param mask: mask of the cut item (sw)
    :param points: points of the contours of the item
    :param show: when true it shows the mask before continue the code
    :return: a mask with a circle in white as mask (this function is nice to crop round things like eyes)
    """
    center, radius = cv2.minEnclosingCircle(points)
    y_point = int(center[1] - radius)
    x_point = int(center[0] - radius)
    h_point = int(2 * radius)
    w_point = h_point

    item = cut_item[y_point: y_point + h_point, x_point: x_point + w_point]
    item_mask = mask[y_point: y_point + h_point, x_point: x_point + w_point]

    if show is True:
        cv2.imshow('item', item)
        cv2.imshow('item_mask', item_mask)
        cv2.waitKey(0)
        cv2.destroyWindow('item')
        cv2.destroyWindow('item_mask')

    return item, item_mask


def create_color_mask(src, color):
    """

    :param src: image where should be looked for the color
    :param color: color in BGR format
    :return: a mask with all pixels white where the color value appears
    """
    mask = np.zeros_like(src)
    cor_mask = np.zeros_like(src)
    mask[np.where(np.all(src == color, axis=-1))] = (0, 0, 0)
    mask[np.where(np.all(src != color, axis=-1))] = (255, 255, 255)

    cor_mask[np.where(np.all(src == color, axis=-1))] = (255, 255, 255)
    cor_mask[np.where(np.all(src != color, axis=-1))] = (255, 255, 255)

    mask = cv2.add(mask, cv2.bitwise_not(cor_mask))

    return mask


def replace(src, face_object, face_object_mask, dst, draw_contours=False):
    """

    :param src: source image from camara ore costume
    :param face_object:  object which should be replaced
    :param face_object_mask:  mask of the face object for overlaying
    :param dst:  x, y coordinates of the face object
    :param draw_contours:  when contours should be draw set that to Ture (if unsure try it)
    :return:  a new image with the replaced object
    """
    new_image = src.copy()
    same_size_mask = np.zeros_like(src)
    same_size_object = np.zeros_like(src)

    '# Anti-aliasing'
    kernel = np.ones((5, 5)) / 25.0
    face_object_kernel = cv2.filter2D(face_object, -1, kernel)

    same_size_mask[dst[1]: dst[1] + face_object.shape[:2][0], dst[0]: dst[0] + face_object.shape[:2][1]] = \
        face_object_mask

    same_size_object[dst[1]: dst[1] + face_object.shape[:2][0], dst[0]: dst[0] + face_object.shape[:2][1]] = \
        face_object_kernel

    new_image[np.where(same_size_mask == 255)] = same_size_object[np.where(same_size_mask == 255)]
    if draw_contours is True:
        mask_gray = cv2.cvtColor(same_size_mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cv2.drawContours(new_image, [contours[i]], 0, (0, 0, 0), 2)

    return new_image
