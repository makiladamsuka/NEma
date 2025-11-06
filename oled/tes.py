import cv2

image = cv2.imread("oled_frames/frame_0000.png")

cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Left OLED View (Rotated)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right OLED View (Rotated)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", 128, 128)
cv2.resizeWindow("Left OLED View (Rotated)", 64, 128)
cv2.resizeWindow("Right OLED View (Rotated)", 64, 128)

cv2.moveWindow("Original Image", 50, 50)
cv2.moveWindow("Left OLED View (Rotated)", 200, 50)
cv2.moveWindow("Right OLED View (Rotated)", 330, 50)

left_half = image[0:128, 0:64]
right_half = image[0:128, 64:128]


cv2.imshow("Original Image", image)
rotated_left = cv2.rotate(left_half, cv2.ROTATE_90_CLOCKWISE)
rotated_right = cv2.rotate(right_half, cv2.ROTATE_90_CLOCKWISE) 

cv2.imshow("Left OLED View (Rotated)", rotated_left)
cv2.imshow("Right OLED View (Rotated)", rotated_right)

cv2.waitKey(0)
cv2.destroyAllWindows()