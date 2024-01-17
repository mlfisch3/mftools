import cv2
import numpy as np
import pytesseract
import pandas as pd
import argparse

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image

def detect_axes(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    return lines

def read_tick_values(image, axes):
    # Simplified for prototype
    tick_mark_location = (100, 100)  # Placeholder values
    tick_mark_roi = image[tick_mark_location[1]-10:tick_mark_location[1]+10, tick_mark_location[0]-10:tick_mark_location[0]+10]
    tick_value = pytesseract.image_to_string(tick_mark_roi, config='--psm 6')
    return tick_value

def calculate_scale_factor(tick_value, pixel_distance):
    numerical_value = float(tick_value)
    scale_factor = numerical_value / pixel_distance
    return scale_factor

def detect_data_points(image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    return circles

def estimate_data_point_value(circles, scale_factor):
    data_values = []
    for circle in circles[0, :]:
        x, y, _ = circle
        data_value = (x * scale_factor, y * scale_factor)
        data_values.append(data_value)
    return data_values

def export_data(data_values, output_file):
    df = pd.DataFrame(data_values, columns=['X Value', 'Y Value'])
    df.to_csv(output_file, index=False)

def main(image_path, output_file):
    processed_image = load_and_preprocess_image(image_path)
    axes = detect_axes(processed_image)
    tick_value = read_tick_values(processed_image, axes)
    scale_factor_x = calculate_scale_factor(tick_value, 100)  # Placeholder pixel distance
    circles = detect_data_points(processed_image)
    data_values = estimate_data_point_value(circles, scale_factor_x)
    export_data(data_values, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract data points from a plot image.')
    parser.add_argument('image_path', type=str, help='Path to the plot image file.')
    parser.add_argument('output_file', type=str, help='Output CSV file path.')
    args = parser.parse_args()

    main(args.image_path, args.output_file)
