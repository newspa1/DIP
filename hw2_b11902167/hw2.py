import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

class Image:
    def denoise_Guassian(img, kernelsize):
        kernel = np.ones((kernelsize, kernelsize), np.float32) / (kernelsize**2)
        img_new = img.copy()
        for r in range(kernelsize//2, img_new.shape[0] - kernelsize//2):
            for c in range(kernelsize//2, img_new.shape[1] - kernelsize//2):
                sub_img = img_new[r-kernelsize//2:r+kernelsize//2+1, c-kernelsize//2:c+kernelsize//2+1]
                img_new[r, c] = np.sum(sub_img * kernel)

        return img_new

    def compute_manitude_and_orientation(img):
        Gr = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Gc = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

        gradient_magnitude = np.zeros(img.shape, dtype=np.uint8)
        gradient_orientation = np.zeros(img.shape, dtype=np.float64)
        for r in range(1, img.shape[0] - 1):
            for c in range(1, img.shape[1] - 1):
                sub_img = img[r-1:r+2, c-1:c+2]
                gradient_magnitude[r, c] = np.sqrt(np.sum(sub_img * Gr)**2 + np.sum(sub_img * Gc)**2)
                orientation = np.arctan2(np.sum(sub_img * Gc), np.sum(sub_img * Gr))
                gradient_orientation[r, c] = orientation if orientation >= 0 else orientation + np.pi

        return gradient_magnitude, gradient_orientation

    def non_maximum_suppression(img, orientation):
        for r in range(1, img.shape[0]-1):
            for c in range(1, img.shape[1]-1):
                angle = orientation[r, c]
                # print(orientation[r, c])
                if angle < np.pi/8 or angle >= 7*np.pi/8:
                    if img[r, c] < img[r, c-1] or img[r, c] < img[r, c+1]:
                        img[r, c] = 0
                elif angle >= np.pi/8 and angle < 3*np.pi/8:
                    if img[r, c] < img[r-1, c+1] or img[r, c] < img[r+1, c-1]:
                        img[r, c] = 0
                elif angle >= 3*np.pi/8 and angle < 5*np.pi/8:
                    if img[r, c] < img[r-1, c] or img[r, c] < img[r+1, c]:
                        img[r, c] = 0
                elif angle >= 5*np.pi/8 and angle < 7*np.pi/8:
                    if img[r, c] < img[r-1, c-1] or img[r, c] < img[r+1, c+1]:
                        img[r, c] = 0
        
        return img

    def finding_connected(img):
        img_new = img.copy()
        stack = []

        # DFS finding connected component
        for r in range(1, img_new.shape[0]-1):
            for c in range(1, img_new.shape[1]-1):
                if img_new[r, c] == 255:
                    stack.append((r, c))
                    while stack:
                        r, c = stack.pop()
                        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                        for dr, dc in directions:
                            if 0 < img_new[r+dr, c+dc] < 255:
                                stack.append((r+dr, c+dc))
                        img_new[r, c] = 255
        
        # Remove not connected component
        img_new[img_new < 255] = 0
        return img_new
            

    def Canny(img, threshold1, threshold2):
        img_new = Image.denoise_Guassian(img, 5)
        img_new, orientation = Image.compute_manitude_and_orientation(img_new)
        
        # # Non-maximum suppression
        img_new = Image.non_maximum_suppression(img_new, orientation)

        # Hysteretic thresholding
        img_new[img_new < threshold1] = 0
        img_new[img_new >= threshold2] = 255

        # Connected component labeling
        img_new = Image.finding_connected(img_new)
        
        return img_new

    def Sobel(img):
        return Image.compute_manitude_and_orientation(img)[0] 

    def thresholding(img, threshold):
        img_new = img.copy()
        img_new[img_new < threshold] = 0
        img_new[img_new >= threshold] = 255
        return img_new

    def shift(img):
        shift_value = np.min(img)
        if shift_value < 0:
            img += abs(shift_value)
        return np.clip(img, 0, 255).astype(np.uint8)


    def convolution(img, kernel):
        img_new = np.zeros(img.shape, dtype=np.float64)
        for r in range(1, img.shape[0]-1):
            for c in range(1, img.shape[1]-1):
                sub_img = img[r-1:r+2, c-1:c+2]
                img_new[r, c] = np.sum(sub_img * kernel)
        
        return img_new

    def Laplacian_of_Gaussian(img, threshold):
        def check_edge(img_new, r, c):
            directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
            for dr, dc in directions:
                if img_new[r+dr, c+dc] * img_new[r-dr, c-dc] < 0:
                    return True
            return False
    
        # Gaussian filter
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
        img_new = Image.convolution(img, kernel)
        img_new = Image.convolution(img_new, kernel)
        # img_new = Image.convolution(img_new, kernel)

        # Laplacian filter
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 4
        img_new = Image.convolution(img_new, kernel)

        # plt.hist(img_new.ravel(), 256)
        # plt.savefig('test/result4_1')
        # Zero-crossing
        img_new[np.abs(img_new) < threshold] = 0

        # Edge detection
        edge_map = np.zeros(img_new.shape, dtype=np.uint8)
        for r in range(1, img_new.shape[0]-1):
            for c in range(1, img_new.shape[1]-1):
                if img_new[r, c] == 0 and check_edge(img_new, r, c):
                    edge_map[r, c] = 255
                else:
                    edge_map[r, c] = 0

        return edge_map
    
    def hough_transform(img):
        max_rho = int(np.sqrt(img.shape[0]**2 + img.shape[1]**2))
        hough = np.zeros((180, max_rho))
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                if img[r, c] == 255:
                    for theta in range(180):
                        rho = int(r * np.cos(np.deg2rad(theta-90)) + c * np.sin(np.deg2rad(theta-90)))
                        hough[theta, rho] += 1
        return hough

    def overlay_lines(img, hough, threshold):
        result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Find the lines
        lines = []
        for theta in range(hough.shape[0]):
            for rho in range(hough.shape[1]):
                if hough[theta, rho] >= threshold:
                    lines.append((theta, rho))
        
        # Overlay the lines
        for theta, rho in lines:
            if 45 < theta < 135:
                # Horizontal line
                for c in range(img.shape[1]):
                    if np.sin(np.deg2rad(theta-90)) != 0:
                        r = int((rho - c * np.sin(np.deg2rad(theta-90))) / np.cos(np.deg2rad(theta-90)))
                        if 0 <= r < img.shape[0]:
                            result[r, c] = 255
            else :
                # Vertical line
                for r in range(img.shape[0]):
                    if np.sin(np.deg2rad(theta-90)) != 0:
                        c = int((rho - r * np.cos(np.deg2rad(theta-90))) / np.sin(np.deg2rad(theta-90)))
                        if 0 <= c < img.shape[1]:
                            result[r, c] = 255

        return result
    
    def Image2Cartesian(img):
        result = np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8)
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                result[x, y] = img[img.shape[0]-y-1, x]
        return result
    
    def Cartesian2Image(img):
        result = np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8)
        for r in range(img.shape[1]):
            for c in range(img.shape[0]):
                result[r, c] = img[c, img.shape[1]-r-1]
        return result

    def matrix_transfer(img_cartesian, matrix):
        matrix = np.linalg.inv(matrix)
        result = np.full((img_cartesian.shape[0], img_cartesian.shape[1]), 255, dtype=np.uint8)
        for x in range(img_cartesian.shape[0]):
            for y in range(img_cartesian.shape[1]):
                transfer = np.matmul(matrix, np.array([x, y, 1]))
                x_old, y_old = int(transfer[0]), int(transfer[1])
                if 0 <= x_old < img_cartesian.shape[0] and 0 <= y_old < img_cartesian.shape[1]:
                    result[x, y] = img_cartesian[x_old, y_old]
        return result

    def translation(img, dx, dy):
        img_cartesian = Image.Image2Cartesian(img)
        matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        result = Image.matrix_transfer(img_cartesian, matrix)
        return Image.Cartesian2Image(result)

    def scaling(img, center, scalex, scaley):
        img_cartesian = Image.Image2Cartesian(img)
        matrix_translate = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
        matrix_scale = np.array([[scalex, 0, 0], [0, scaley, 0], [0, 0, 1]])
        matrix_back = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
        matrix = np.matmul(matrix_scale, matrix_translate)
        matrix = np.matmul(matrix_back, matrix)
        result = Image.matrix_transfer(img_cartesian, matrix)
        return Image.Cartesian2Image(result)

    def rotation(img, angle, pivot=None):
        img_cartesian = Image.Image2Cartesian(img)

        # Rotate by pivot
        if pivot is None:
            pivot = (img.shape[0]//2, img.shape[1]//2)
        
        matrix_translate = np.array([[1, 0, -pivot[0]], [0, 1, -pivot[1]], [0, 0, 1]])
        matrix_rotate = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
                                    [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0],
                                    [0, 0, 1]])
        matrix_back = np.array([[1, 0, pivot[0]], [0, 1, pivot[1]], [0, 0, 1]])
        matrix = np.matmul(matrix_rotate, matrix_translate)
        matrix = np.matmul(matrix_back, matrix)
        
        result = Image.matrix_transfer(img_cartesian, matrix)
        return Image.Cartesian2Image(result)

    def distortion(img, center, k1, k2):
        img_cartesian = Image.Image2Cartesian(img)

        result = np.full((img_cartesian.shape[0], img_cartesian.shape[1]), 255, dtype=np.uint8)
        for x_new in range(img_cartesian.shape[0]):
            for y_new in range(img_cartesian.shape[1]):
                r_new = np.sqrt((x_new - center[0])**2 + (y_new - center[1])**2)
                r_old = r_new * (1 + k1 * r_new**2 + k2 * r_new**4)

                x_old = int(center[0] + (x_new - center[0]) * r_old / (r_new+0.1))
                y_old = int(center[1] + (y_new - center[1]) * r_old / (r_new+0.1))

                if 0 <= x_old < img_cartesian.shape[0] and 0 <= y_old < img_cartesian.shape[1]:
                    result[x_new, y_new] = img_cartesian[x_old, y_old]
                
        return Image.Cartesian2Image(result)

    def swirling(img, center, radius, strength, rotation):
        img_cartesian = Image.Image2Cartesian(img)
        result = img_cartesian.copy()
        
        for x in range(img_cartesian.shape[0]):
            for y in range(img_cartesian.shape[1]):
                if np.sqrt((center[0]-x)**2 + (center[1]-y)**2) < radius:
                    theta_new = np.arctan2(x - center[0], y - center[1])
                    rho = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    r = np.log(2) * radius / 5
                    angle  = strength * np.exp(-rho/r)
                    
                    theta_old = rotation + angle + theta_new
                    x_old = int(center[0] + rho * np.sin(theta_old))
                    y_old = int(center[1] + rho * np.cos(theta_old))
                    if 0 <= x_old < img_cartesian.shape[0] and 0 <= y_old < img_cartesian.shape[1]:
                        result[x, y] = img_cartesian[x_old, y_old]

        return Image.Cartesian2Image(result)
        


def problem1_a():
    sample1 = cv2.imread('hw2_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    result1 = Image.Sobel(sample1)
    result2 = Image.thresholding(result1, 95)
    cv2.imwrite('result1.png', result1)
    cv2.imwrite('result2.png', result2)

    # # Different threshold
    # result2_1 = Image.thresholding(result1, 50)
    # cv2.imwrite('test/result2_1.png', result2_1)
    # result2_2 = Image.thresholding(result1, 120)
    # cv2.imwrite('test/result2_2.png', result2_2)

def problem1_b():
    sample1 = cv2.imread('hw2_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    result3 = Image.Canny(sample1, 10, 90)
    cv2.imwrite('result3.png', result3)

    # # Different threshold
    # result3_1 = Image.Canny(sample1, 10, 120)
    # cv2.imwrite('test/result3_1.png', result3_1)
    # result3_2 = Image.Canny(sample1, 5, 90)
    # cv2.imwrite('test/result3_2.png', result3_2)
    # result3_3 = Image.Canny(sample1, 30, 60)
    # cv2.imwrite('test/result3_3.png', result3_3)

def problem1_c():
    sample1 = cv2.imread('hw2_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    result4 = Image.Laplacian_of_Gaussian(sample1, 0.8)
    cv2.imwrite('result4.png', result4)

def problem1_d():
    sample1 = cv2.imread('hw2_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    portrait = Image.Sobel(sample1)
    portrait_1 = Image.thresholding(portrait, 180)
    cv2.imwrite('test/portrait.png', portrait_1)

    # # Different threshold
    # portrait_2 = Image.thresholding(portrait, 200)
    # cv2.imwrite('test/portrait_2.png', portrait_2)

def problem1_e():
    sample2 = cv2.imread('hw2_sample_images/sample2.png', cv2.IMREAD_GRAYSCALE)
    result5 = Image.Canny(sample2, 60, 70)
    cv2.imwrite('result5.png', result5)

    # Hough transform
    result6 = Image.hough_transform(result5)
    plt.imshow(result6, cmap='gray', aspect='auto', origin='lower')
    plt.xlabel('rho')
    plt.ylabel('theta (degree)')
    plt.savefig('result6.png')

    # overlay the lines
    result7 = Image.overlay_lines(result5, result6, 80)
    cv2.imwrite('result7.png', result7)

def problem2_a():
    sample3 = cv2.imread('hw2_sample_images/sample3.png', cv2.IMREAD_GRAYSCALE)
    result8 = sample3.copy()
    result8 = Image.distortion(result8, center=(300, 300), k1=0.0001, k2=0)
    # cv2.imwrite('test/distortion.png', result8)
    result8 = Image.translation(result8, -70, 60)
    result8 = Image.rotation(result8, 15)
    result8 = Image.scaling(result8, center=(300, 300), scalex=2, scaley=2)

    cv2.imwrite('result8.png', result8)

def problem2_b():
    sample5 = cv2.imread('hw2_sample_images/sample5.png', cv2.IMREAD_GRAYSCALE)
    result9 = sample5.copy()
    result9 = Image.swirling(result9, center=(200, 450), radius=300, strength=24, rotation=0)
    cv2.imwrite('result9.png', result9)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("choice", type=int, choices=range(1, 8))
    args = parser.parse_args()

    options = {
        1: problem1_a,
        2: problem1_b,
        3: problem1_c,
        4: problem1_d,
        5: problem1_e,
        6: problem2_a,
        7: problem2_b,
    }

    options[args.choice]()
    # plt.show()
