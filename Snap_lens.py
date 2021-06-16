import numpy as np
import snap_lib
import cv2

cam = cv2.VideoCapture(0)

spongebob = cv2.imread('Spongebob.png')
village = cv2.imread('village.png')

'# set coordinates fro the positions of the face objects'
spongebob_left_eye = (140, 55)
spongebob_right_eye = (222, 55)
spongebob_mouth = (130, 170)


while cam.isOpened():
    _, frame = cam.read()
    new_background_image = village.copy()
    # shape frame in same size as the background image (village)
    resized_frame = snap_lib.resize_image(frame, (village.shape[1], village.shape[0]))

    # detect faces on the resized frame
    faces = snap_lib.face_detection(resized_frame, 1)
    # shown_faces = snap_lib.show_faces(resized_frame, faces)

    # detect landmark points on the resized frame
    face_landmarks = snap_lib.get_landmarks(resized_frame, faces)

    # separate mouth and eyes from the src image
    mouth_and_eye_mask = snap_lib.get_eyes_and_mouth_mask(resized_frame, face_landmarks)
    cut_mouth_and_eyes = snap_lib.cut_out_mask(resized_frame, mouth_and_eye_mask)

    for face in face_landmarks:
        mouth, left_eye, right_eye = snap_lib.get_mouth_and_eye_points(face)
        cut_mouth, mouth_mask = snap_lib.cut_rectangle_from_points(cut_mouth_and_eyes, mouth_and_eye_mask,
                                                                   mouth, show=False)
        cut_left_eye, left_eye_mask = snap_lib.cut_circle_from_points(cut_mouth_and_eyes, mouth_and_eye_mask, left_eye,
                                                                      show=False)

        cut_right_eye, right_eye_mask = snap_lib.cut_circle_from_points(cut_mouth_and_eyes, mouth_and_eye_mask,
                                                                        right_eye, show=False)

        '# Place mouth and eyes on spongebob'
        item_and_point_list = [(cut_mouth, mouth_mask, spongebob_mouth),
                               (cut_left_eye, left_eye_mask, spongebob_left_eye),
                               (cut_right_eye, right_eye_mask, spongebob_right_eye)]
        # create new spongebob image
        new_spongebob = spongebob.copy()
        # resizing values
        factor_x = 2.5
        factor_y = 2.5
        for counter, (item, mask, point) in enumerate(item_and_point_list):
            # resize_item
            if counter == 1:
                factor_x += 0.2
                factor_y += 0.2
            bigger_item = snap_lib.resize_image(item, (int(item.shape[1] * factor_x), int(item.shape[0] * factor_y)))
            bigger_mask = snap_lib.resize_image(mask, (int(mask.shape[1] * factor_x), int(mask.shape[0] * factor_y)))
            new_spongebob = snap_lib.replace(new_spongebob, bigger_item, bigger_mask, point)

        '# place_spongebob on background_image'
        # 1. resize spongebob to same size as the detected face
        x_pos, y_pos, w, h = cv2.boundingRect(np.array(face, np.int32))
        resized_spongebob = snap_lib.resize_image(new_spongebob, (w, h))

        '# place spongebob on a black image, which has the same size as the village'
        spongebob_same_size = np.zeros_like(village)
        spongebob_same_size[0: spongebob_same_size.shape[0], 0: spongebob_same_size.shape[1]] = (0, 255, 0)
        try:
            spongebob_same_size[y_pos: y_pos + h, x_pos: x_pos + w] = resized_spongebob
        except ValueError:
            continue

        '# replace spongebob on background'
        # 1. create mask
        mask = snap_lib.create_color_mask(spongebob_same_size, (0, 255, 0))
        # 2. replace spongebob
        new_background_image[np.where(np.all(mask == (255, 255, 255), axis=-1))] = \
            spongebob_same_size[np.where(np.all(mask == (255, 255, 255), axis=-1))]

    cv2.imshow('new_background_image', new_background_image)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
