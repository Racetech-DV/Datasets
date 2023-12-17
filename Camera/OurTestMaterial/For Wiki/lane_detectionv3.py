import cv2
import numpy as np
import time
from skimage.morphology import skeletonize
import skimage
from timeit import default_timer as timer
import math
import os
FIVE_DEGREES_IN_RAD = 0.0873
TAN_OF_TWENTY_DEGREES = 0.364
RIGHT_REGION = 50
LEFT_REGION = -50
GO_LEFT = -23
GO_STRAIGHT = 0
GO_RIGHT = 23
STOP_LINE = 40
PARK_LINE = -49

STOP_LINE_DISTANCE_TABLE = [
    [1, 10],
    [2, 30],
    [3, 50],
    [4, 60],
    [5, 65]
]


def process_img(img):
    img = img[int(2 * img.shape[0] / 3):img.shape[0], :]
    canny_min_thresh = 100
    canny_max_thresh = 240
    img = cv2.Canny(img, canny_min_thresh, canny_max_thresh)
    width = int(img.shape[1] / 2)
    height = int(img.shape[0] / 2)
    dim = (width, height)
    img = cv2.resize(img, dim, cv2.INTER_AREA)
    img[img > 0] = 255
    global _center_col
    _center_col = (img.shape[1] - 1) / 2
    cv2.imwrite(os.path.abspath(os.path.join(os.getcwd(), 'edges.jpg')), img)
    img = skeletonize(img / 255)
    img = img.astype(np.uint8) * 255
    return img


def find_slope(line):
    dx = line[2]-line[0]
    dy = line[3]-line[1]
    if dx == 0:
        return 1000*dy
    return (line[3]-line[1])/(line[2]-line[0])


def find_stopping_lines(lines):
    for i, line1 in enumerate(lines):
        slope1 = find_slope(line1)
        if math.fabs(slope1) < TAN_OF_TWENTY_DEGREES: # if it is maximum 20 degrees sloped
            for j, line2 in enumerate(lines[i+1:], i+1):
                slope2 = find_slope(line2)
                angle1 = math.atan(slope1)
                angle2 = math.atan(slope2)
                if math.fabs(angle1-angle2) < FIVE_DEGREES_IN_RAD: # if two lines are parallel
                    intercept1 = -slope1*line1[0]+line1[1]
                    intercept2 = -slope1 * line2[0] + line2[1]
                    if math.fabs(intercept2-intercept1) < 25: # if two lines are near
                        return line1, line2
    else:
        return None, None


def find_stopping_lane(img):
    mid_row = 0
    stop_lane_frame = np.copy(img)
    lines = cv2.HoughLinesP(img, 1, np.pi / 720, threshold=80, minLineLength=100, maxLineGap=30)
    if lines is None:
        return mid_row, stop_lane_frame
    lines = lines[:, 0].tolist()
    order_lines(lines)
    for l in lines:
        cv2.line(stop_lane_frame, (l[0], l[1]), (l[2], l[3]), 100, 2, cv2.LINE_AA)
    line1, line2 = find_stopping_lines(lines)
    if line1 is None:
        return mid_row, stop_lane_frame
    mid_row = int((line1[1]+line1[3]+line2[1]+line2[3])/4)
    cv2.line(stop_lane_frame, (line1[0], line1[1]), (line1[2], line1[3]), 255, 4, cv2.LINE_AA)
    cv2.line(stop_lane_frame, (line2[0], line2[1]), (line2[2], line2[3]), 255, 4, cv2.LINE_AA)
    return mid_row, stop_lane_frame


def find_lanes(img):
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, threshold=4, minLineLength=10, maxLineGap=5)
    if lines is None:
        return img
    lines = lines[:,0].tolist()
    order_lines(lines)
    # FOR DEBUGGING #
    lines_frame = np.copy(img)
    lines_frame = np.stack([lines_frame, lines_frame, lines_frame], axis = 2)
    for l in lines:
        cv2.line(lines_frame, (l[0], l[1]), (l[2], l[3]), (200, 200, 0), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.abspath(os.path.join(os.getcwd(), 'lines.jpg')), lines_frame)
    cv2.imshow("lines", lines_frame)
    lines_frame = np.copy(img)
    lines_frame = np.stack([lines_frame, lines_frame, lines_frame], axis=2)
    # END FOR DEBUGGING #
    left_lane = find_left_lane(lines, LEFT_REGION, img)
    right_lane = find_left_lane(lines, RIGHT_REGION, img)
    if len(left_lane)==0:
       left_lane = None
    if len(right_lane)==0:
        right_lane = None
    #### FOR DEBUGGING ####

    if left_lane is not None:
        for line in left_lane:
            cv2.line(lines_frame, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2, cv2.LINE_AA)
    if right_lane is not None:
       for line in right_lane:
           cv2.line(lines_frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2, cv2.LINE_AA)
    #### END FOR DEBUGGING ####
    return left_lane, right_lane, lines_frame


def swap_start_end(line):
    temp_start_x = line[2]
    temp_start_y = line[3]
    line[2] = line[0]  # end x
    line[3] = line[1]  # end y
    line[0] = temp_start_x  # start x
    line[1] = temp_start_y  # start y


def order_lines(lines):
    for line in lines:
        if line[3] > line[1]: # if end is lower than start, swap start and end
            swap_start_end(line)
        elif line[3] == line[1]: # if end is at the same height as start
            if line[2] < line[0]: # if end is lefter than start
                swap_start_end(line)


def is_brother(line1, line2): # returns line 2 if it is brother of line 1, otherwise None, line 1 is not brother
    if line2[3] >= line1[3]: # if second ends lower than first
        return None
    if line2[1] + 10 < line1[3]: # if second starts more than 5 pixels higher than first ends
        return None
    if math.fabs(cv2.pointPolygonTest(np.array([[line2[0], line2[1]], [line2[2], line2[3]]], dtype = np.float32),
                            (int(line1[2]), int(line1[3])), True)) > 13: # if distance between end of first and second is bigger than 7
        return None
    slope1 = find_slope(line1)
    slope2 = find_slope(line2)
    angle1 = math.atan(math.fabs(slope1)) # absolute angle
    angle2 = math.atan(math.fabs(slope2))
    if math.fabs(angle2-angle1) > np.pi/6: # if angle between lines is more than 30 degrees
        return None
    return line2


def find_left_lane(lines, search_region, img):
    i = 0
    left_lane = []
    prev_first = None
    curr_line = find_first_line(prev_first, lines, search_region, img) # start with some line, returns None if not found
    prev_first = curr_line
    next_line = curr_line
    while next_line is not None:
        left_lane.append(next_line)
        for line in lines: # search for next line (brother)
            if search_region == RIGHT_REGION:
                img = cv2.line(img, (curr_line[0], curr_line[1]), (curr_line[2], curr_line[3]), (50 + i, 0, 0), 5)
                cv2.imshow("sequence", img)
                cv2.imwrite(os.path.abspath(os.path.join(os.getcwd(), 'sequence.jpg')), img)
            next_line = is_brother(curr_line, line) # curr_line will be None if 'line' is not brother
            if next_line is not None: # if found next line go further
                if search_region == RIGHT_REGION:
                    i += 50
                curr_line = next_line
                break
    return left_lane


def find_ref_col(row, height): # returns 20 if row = height-1, 10 if row = 0, linear interpolation between
    a = (20-10)/(height-1)
    b = 10
    return round(a*row+b)


def adjust_col_ind(col_ind, width):
    if col_ind<0:
        return 0
    if col_ind>width-1:
        return width-1
    else:
        return col_ind


def check_brotherhood(line, img): # returns True if there are whites in specific range in at least two of: start, end, mid
    start_row = line[1]
    start_col = line[0]
    end_row = line[3]
    end_col = line[2]
    mid_col, mid_row = midpoint(start_col, start_row, end_col, end_row)
    height = img.shape[0]
    width = img.shape[1]
    start_ref_col = find_ref_col(start_row, height)
    mid_ref_col = find_ref_col(mid_row, height)
    end_ref_col = find_ref_col(end_row, height)
    are_whites_near_start = 1 if np.sum(img[start_row, adjust_col_ind(start_col - start_ref_col, width):adjust_col_ind(start_col + start_ref_col, width)]) > 255 else 0
    are_whites_near_end = 1 if np.sum(img[end_row, adjust_col_ind(end_col - end_ref_col, width):adjust_col_ind(end_col + end_ref_col, width)]) > 255 else 0
    are_whites_near_mid = 1 if np.sum(img[mid_row, adjust_col_ind(mid_col - mid_ref_col, width):adjust_col_ind(mid_col + mid_ref_col, width)]) > 255 else 0
    if are_whites_near_start+are_whites_near_end+are_whites_near_mid < 2:
        return False
    else:
        return True


def find_first_line(prev_first, lines, search_region, img):
    first_line = None
    lowest_y = 0
    width = int(img.shape[1] / 2)
    height = int(img.shape[0] / 2)
    for line in lines:
        start_y = line[1]
        if start_y < lowest_y:  # if line starts higher than lowest_y
            continue
        if (line[1]<height/2):# if it starts in the top half
            continue
        if (search_region == RIGHT_REGION and line[0] < _center_col
                or search_region == LEFT_REGION and line[0] > _center_col): # if line starts not in the left(right) half
            continue
        if prev_first is not None: # distance in columns between it and previous is less than third of image
            if math.fabs(prev_first[0]-line[0])<width/3:
                continue
        slope = math.fabs(find_slope(line))
        angle = math.atan(slope)
        if angle < np.pi/6 : # if angle of line is not between 45 and 135 degrees
            continue
        if not check_brotherhood(line,img): # if there are no whites in specific range near start, mid, end
            continue
        lowest_y = start_y  # else update first_line
        first_line = line
    return first_line


def find_park_lane(img):
    stop_lane_frame = np.copy(img)
    lines = cv2.HoughLinesP(img, 1, np.pi / 720, threshold=40, minLineLength=50, maxLineGap=15)
    if lines is None:
        return stop_lane_frame
    lines = lines[:, 0].tolist()
    order_lines(lines)
    for l in lines:
        cv2.line(stop_lane_frame, (l[0], l[1]), (l[2], l[3]), 150, 2, cv2.LINE_AA)
    line1, line2 = find_stopping_lines(lines)
    if line1 is None:
        return stop_lane_frame
    cv2.line(stop_lane_frame, (line1[0], line1[1]), (line1[2], line1[3]), 200, 4, cv2.LINE_AA)
    cv2.line(stop_lane_frame, (line2[0], line2[1]), (line2[2], line2[3]), 200, 4, cv2.LINE_AA)
    return stop_lane_frame


def midpoint(start_x, start_y, end_x, end_y):
    return round((start_x+end_x)/2), round((start_y+end_y)/2)


def give_lane_keeping_angle(mid_row, left_lane, right_lane, img): # left_lane, right_lane are list of lines each line is list [x1,y1,x2,y2]
    left_steering_angle = right_steering_angle = GO_STRAIGHT
    if left_lane is not None:
        lane_points = [[point[0], point[1]] for row in left_lane for point in zip(row[::2], row[1::2])]
        rows = [point[1] for point in lane_points]
        cols = [point[0] for point in lane_points]
        lane_coeff, residuals, _, _, _ = np.polyfit(rows, cols, 2, full=True)
        #print(lane_coeff)
        top_intersection_x = np.polyval(lane_coeff, -mid_row)
        if top_intersection_x >= img.shape[1]/2:
            left_steering_angle = GO_RIGHT
        #print('Top intersection: %d' %(top_intersection_x))
    #print('Now right lane:')
    if right_lane is not None:
        lane_points = [[point[0], point[1]] for row in right_lane for point in zip(row[::2], row[1::2])]
        rows = [point[1] for point in lane_points]
        cols = [point[0] for point in lane_points]
        lane_coeff, residuals, _, _, _ = np.polyfit(rows, cols, 2, full=True)
        #print(lane_coeff)
        top_intersection_x = np.polyval(lane_coeff, -mid_row)
        if top_intersection_x <= img.shape[1]/2:
            left_steering_angle = GO_LEFT
        #print('Top intersection: %d' %(top_intersection_x))
        #### FOR DEBUGGING ####
        #### END FOR DEBUGGING ####
        if math.fabs(left_steering_angle)>right_steering_angle:
            return left_steering_angle
        else:
            return right_steering_angle


def find_distance(mid_row, line):
    distance_table = STOP_LINE_DISTANCE_TABLE
    #if line == PARK_LINE:
    #    distance_table = PARK_LINE_DISTANCE_TABLE
    lower_point_x, lower_point_y = max([point for point in distance_table if point[0] <= mid_row]) # find nearest point lower
    upper_point_x, upper_point_y= min([point for point in distance_table if point[0] >= mid_row]) # find nearest point upper
    slope = find_slope([lower_point_x, lower_point_y, upper_point_x, upper_point_y])
    if lower_point_x == mid_row:
        return lower_point_y
    if upper_point_x == mid_row:
        return upper_point_y
    intersept = lower_point_y-slope*lower_point_x
    distance = slope*mid_row+intersept
    return distance


def understanding_lanes(img):
    img = process_img(img)
    cv2.imwrite(os.path.abspath(os.path.join(os.getcwd(), 'processed.jpg')), img)
    cv2.imshow("process_img", img)
    mid_row, stop_lane_frame = find_stopping_lane(img)
    left_lane, right_lane, lanes_frame = find_lanes(img[mid_row:img.shape[0],:]) # left_lane, right_lane are list of lines.
                                                         # each line is list [x1,y1,x2,y2]
    lane_keeping_angle = give_lane_keeping_angle(mid_row, left_lane,  right_lane, lanes_frame)
    park_lane_frame = find_park_lane(img)
    #(find_distance(5, STOP_LINE))
    return lanes_frame, stop_lane_frame


imgpath = os.path.abspath(os.path.join(os.getcwd(), '..', 'Test_13_02_23_Frames/6510.jpg'))
print(imgpath)
img = cv2.imread(imgpath)
cv2.imshow("input", img)
lanes_frame, stop_lane_frame = understanding_lanes(img)
cv2.imwrite(os.path.abspath(os.path.join(os.getcwd(), 'output.jpg')), lanes_frame)
cv2.imwrite(os.path.abspath(os.path.join(os.getcwd(), 'output_stopq.jpg')), stop_lane_frame)
cv2.imshow("output", lanes_frame)
cv2.imshow("output_stop", stop_lane_frame)
cv2.waitKey(0)


