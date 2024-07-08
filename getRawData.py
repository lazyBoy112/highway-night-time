import cv2
import os
path = "D:/sang/project/python/highway-night-time/rawdata/video/1.mp4"

def count_files_in_directory(directory):
    try:
        # Lấy danh sách tất cả các file và thư mục trong thư mục đã chỉ định
        files_and_dirs = os.listdir(directory)
        # Lọc ra chỉ các file
        files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))]
        # Đếm số lượng file
        return len(files)
    except Exception as e:
        print(f"error: {e}")
        return 0

positive_id = count_files_in_directory('positive')
negative_id = count_files_in_directory('negative')
cap = cv2.VideoCapture(path)


if (not cap.isOpened()):
    print('error when open video')

while(cap.isOpened()):

    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('proprocess to negative and positive', gray_frame)

    # waits 1 ms every loop to process key presses
    key = cv2.waitKey(20)
    if key == ord('p'):
        cv2.imwrite('positive\\{}.bmp'.format('{:04}'.format(positive_id)), gray_frame)
        positive_id = positive_id + 1
    elif key == ord('n'):
        cv2.imwrite('negative\\{}.bmp'.format('{:04}'.format(negative_id)), gray_frame)
        negative_id = negative_id + 1
    elif key == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()

