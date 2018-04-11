import calibration
import lane_finding
import cv2
import visualizer

image_size = (1280,720)
# PREPROCESSING: CAMERA CALIBRATION
cal_mtx,dist_coeff = calibration.calibrate()
# print(cal_mtx)
# print(dist_coeff)

# PROCESSING:
lane_finder = lane_finding.LaneFinder(cal_mtx,dist_coeff,image_size)

# ON TEST IMAGE
lane_finding.debug = False
# test_image = cv2.imread('test_images/test5.jpg')
# # visualizer.plot_colorspaces(test_image,'output_images/test_image/')
# # lane_finder.adjust_parameters(test_image)
# test_result = lane_finder.process_image(test_image)

# ON PROJECT VIDEO
lane_finding.debug = False
cap = cv2.VideoCapture('project_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_video.mp4',fourcc, 60.0, image_size)

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret==True:
        i += 1
        #print("Frame " + str(i))
        result = lane_finder.process_image(frame)
        out.write(result)

        # cv2.imshow('frame2',frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# visualizer.plot_history(lane_finder)