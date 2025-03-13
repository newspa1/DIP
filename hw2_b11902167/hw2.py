import matplotlib.pyplot as plt
import numpy as np
import cv2

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
        # Gc = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        gradient_magnitude = np.zeros(img.shape, dtype=np.uint8)
        gradient_orientation = np.zeros(img.shape, dtype=np.float64)
        for r in range(1, img.shape[0] - 1):
            for c in range(1, img.shape[1] - 1):
                sub_img = img[r-1:r+2, c-1:c+2]
                gradient_magnitude[r, c] = np.sqrt(np.sum(sub_img * Gr)**2 + np.sum(sub_img * Gc)**2)
                orientation = np.arctan2(np.sum(sub_img * Gc), np.sum(sub_img * Gr))
                gradient_orientation[r, c] = orientation if orientation >= 0 else orientation + np.pi

        return gradient_magnitude, gradient_orientation

    def non_maximum_suppression(img, magnitude, orientation):
        for r in range(1, img.shape[0]-1):
            for c in range(1, img.shape[1]-1):
                angle = orientation[r, c]
                # print(orientation[r, c])
                if angle < np.pi/8 or angle >= 7*np.pi/8:
                    if magnitude[r, c] < magnitude[r, c-1] or magnitude[r, c] < magnitude[r, c+1]:
                        img[r, c] = 0
                elif angle >= np.pi/8 and angle < 3*np.pi/8:
                    if magnitude[r, c] < magnitude[r-1, c+1] or magnitude[r, c] < magnitude[r+1, c-1]:
                        img[r, c] = 0
                elif angle >= 3*np.pi/8 and angle < 5*np.pi/8:
                    if magnitude[r, c] < magnitude[r-1, c] or magnitude[r, c] < magnitude[r+1, c]:
                        img[r, c] = 0
                elif angle >= 5*np.pi/8 and angle < 7*np.pi/8:
                    if magnitude[r, c] < magnitude[r-1, c-1] or magnitude[r, c] < magnitude[r+1, c+1]:
                        img[r, c] = 0
        
        return img

    def Canny(img, threshold1, threshold2):
        img_new = Image.denoise_Guassian(img, 5)
        magnitude, orientation = Image.compute_manitude_and_orientation(img_new)
        directions = [[0, 1], [1, 1], [1, 0], [1, -1]]
        plt.hist(magnitude.ravel())
        # Non-maximum suppression
        img_new = Image.non_maximum_suppression(img_new, magnitude, orientation)

        # Hysteretic thresholding
        # img_new[img_new < threshold1] = 0
        # img_new[img_new >= threshold2] = 255

        # Connected component labeling
        # for r in range(1, img_new.shape[0]-1):
        #     for c in range(1, img_new.shape[1]-1):
        #         if img_new[r, c] != 0:
        #             for direction in directions:
        #                 if img_new[r+direction[0], c+direction[1]] == 255:
        #                     img_new[r, c] = 255
        #                     break
        
        return img_new

    def Sobel(img):
        return Image.compute_manitude_and_orientation(img)[0] 

    def thresholding(img, threshold):
        img_new = img.copy()
        img_new[img_new < threshold] = 0
        img_new[img_new >= threshold] = 255
        return img_new

if __name__ == "__main__":
    sample1 = cv2.imread('hw2_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    # result1 = Image.Sobel(sample1)
    # result2 = Image.thresholding(result1, 45)
    
    # cv2.imwrite('result1.png', result1)
    # cv2.imwrite('result2.png', result2)
    # plt.hist(result1.ravel(), 256, [0, 256])

    result3 = Image.Canny(sample1, 90, 180)
    cv2.imwrite('result3.png', result3)


    plt.show()