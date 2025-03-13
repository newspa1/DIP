import cv2
import numpy as np
import matplotlib.pyplot as plt

class Denoise:
    def Guassian(image, alpha):
        image_new = Image(np.zeros_like(image.img))
        size = (image.img.shape[0], image.img.shape[1])
        matrix = np.array([[1, alpha, 1], [alpha, alpha**2, alpha], [1, alpha, 1]]) / (alpha + 2)**2
            
        def computeMask(r, c):
            top, bottom, left, right = max(0, r - 1), min(size[0] - 1, r + 1), max(0, c - 1), min(size[1] - 1, c + 1)
            pixel_new = round(
                image.img[top][left] * matrix[0][0] + image.img[top][c] * matrix[0][1] + image.img[top][right] * matrix[0][2] + 
                image.img[r][left] * matrix[1][0] + image.img[r][c] * matrix[1][1] + image.img[r][right] * matrix[1][2] + 
                image.img[bottom][left] * matrix[2][0] + image.img[bottom][c] * matrix[2][1] + image.img[bottom][right] * matrix[2][2]
            )
            return pixel_new
            
        for r in range(size[0]):
            for c in range(size[1]):
                image_new.img[r][c] = computeMask(r, c)

        return image_new

    class Impluse:
        def maxmin(pixels):
            maxpixel = pixels[0]
            for index in range(len(pixels) - 2):
                minpixel = min(pixels[index], pixels[index + 1], pixels[index + 2])
                if minpixel > maxpixel:
                    maxpixel = minpixel
            return maxpixel
    
        def minmax(pixels):
            minpixel = pixels[0]
            for index in range(len(pixels) - 2):
                maxpixel = max(pixels[index], pixels[index + 1], pixels[index + 2])
                if maxpixel < minpixel:
                    minpixel = maxpixel
            return minpixel
        
        def getNeighbors(image, r, c, windowsize):
            size = (image.img.shape[0], image.img.shape[1])
            pixels = [image.img[r][c]]
            for i in range(1, windowsize // 2 + 1):
                pixels.append(image.img[max(0, r - i)][c])
                pixels.append(image.img[min(size[0] - 1, r + i)][c])
                pixels.append(image.img[r][max(0, c - i)])
                pixels.append(image.img[r][min(size[1] - 1, c + i)])
            return pixels

        def apply_filter(image, windowsize, type):
            image_new = Image(np.zeros_like(image.img))
            size = (image_new.img.shape[0], image_new.img.shape[1])
            for r in range(size[0]):
                for c in range(size[1]):
                    pixels = Denoise.Impluse.getNeighbors(image, r, c, windowsize)
                    
                    if type == "maxmin":
                        image_new.img[r][c] = Denoise.Impluse.maxmin(pixels)
                    elif type == "minmax":
                        image_new.img[r][c] = Denoise.Impluse.minmax(pixels)
                    elif type == "PMED":
                        image_new.img[r][c] = 0.5 * Denoise.Impluse.maxmin(pixels) + 0.5 * Denoise.Impluse.maxmin(pixels)
            
            return image_new

        def Impluse_maxmin(image, windowsize):
            return Denoise.Impluse.apply_filter(image, windowsize, "maxmin")
        
        def Impluse_minmax(image, windowsize):
            return Denoise.Impluse.apply_filter(image, windowsize, "minmax")

        def Impluse_PMED(image, windowsize):
            return Denoise.Impluse.apply_filter(image, windowsize, "PMED")

    # Calculate mean squared error (MSE)
    def MSE(image_old, image_new):
        size = image_old.size
        sum = 0
        for r in range(size[0]):
            for c in range(size[1]):
                sum += (int(image_old.img[r][c]) - int(image_new.img[r][c]))**2

        return sum / (size[0] * size[1])

    # Calculate peak signal-to-noise ratio (PSNR)
    def PSNR(image_old, image_new):
        mse = Denoise.MSE(image_old, image_new)
        return 10 * np.log10(255**2 / mse)
        
        
class Image:
    def __init__(self, img):
        self.img = img
        self.size = (img.shape[0], img.shape[1])

    # Copy image
    def copy(self):
        return Image(self.img)

    # Horizontal flipped
    def hflip(image):
        return Image(np.flip(image.img, 1))
    
    # Horizontal concatenate two image
    def concatenate(images, type):
        imgs = []
        for image in images:
            imgs.append(image.img)

        if type == 0:
            return Image(np.hstack(tuple(imgs)))
        elif type == 1:
            return Image(np.vstack(tuple(imgs)))

    # Change Brightness
    def changeBrightness(image, value):
        return Image(np.round(image.img * value).astype(np.uint8))
    
    def CDFtransfer_function(img):
        cdf_min = 0
        pixel_min = 0

        # Compute histogram
        histogram = [0] * 256
        for value in img.ravel():
            histogram[value] += 1

        cdf_old = [0] * 256
        flag = True
        for value in range(0, 256):
            cdf_old[value] = cdf_old[max(0, value-1)] + histogram[value]
            if flag and cdf_old[value] != 0:
                cdf_min = cdf_old[value]
                pixel_min = value
                flag = False
        
        # Calculate transfer function
        if len(img) * len(img[0]) - cdf_min == 0:
            return [pixel_min] * 256

        cdf_new = [0] * 256
        for value in range(pixel_min, 256):
            cdf_new[value] = (((cdf_old[value] - cdf_min) / (len(img) * len(img[0]) - cdf_min)) * 255)
        
        return cdf_new

    # Compute global Histogram Equalization
    def histEqualization(image):
        img_new = np.zeros_like(image.img)
        cdf_new = Image.CDFtransfer_function(image.img)
        for row in range(len(image.img)):
            for col in range(len(image.img[0])):
                img_new[row][col] = cdf_new[image.img[row][col]]

        return Image(img_new)
    
    # Concatenate grids back to image
    def gridsConcatenate(grids):
        img = []
        blocksize = (len(grids), len(grids[0]))
        for r in range(blocksize[0]):
            row_grid = grids[r][0]
            for c in range(1, blocksize[1]):
                row_grid = np.hstack((row_grid, grids[r][c]))
            
            if r == 0:
                img = row_grid
            else:
                img = np.vstack((img, row_grid))

        return img
    
    # Smooth the border of images
    def applyCLAHE(img, grids):
        blocksize = (len(grids), len(grids[0]))
        gridsize = (len(grids[0][0]), len(grids[0][0][0]))

        # Calculate transfer function for each grid
        transfer_function = [[0]*blocksize[1] for _ in range(blocksize[0])]
        for r in range(blocksize[0]):
            for c in range(blocksize[1]):
                transfer_function[r][c] = Image.CDFtransfer_function(grids[r][c])

        img_new = np.zeros_like(img)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                tileindex_r, tileindex_c = (r // gridsize[0], c // gridsize[1])
                center_r, center_c = (tileindex_r * gridsize[0] + gridsize[0] / 2, tileindex_c * gridsize[1] + gridsize[1] / 2)
                ratio_r, ratio_c = (1 - abs(r - center_r) / gridsize[0], 1 - abs(c - center_c) / gridsize[1])

                # Find 4 closest grids
                next_r, next_c = (0, 0)
                if r > center_r:
                    next_r = min(tileindex_r + 1, blocksize[0] - 1)
                else:
                    next_r = max(tileindex_r - 1, 0)

                if c > center_c:
                    next_c = min(tileindex_c + 1, blocksize[1] - 1)
                else:
                    next_c = max(tileindex_c - 1, 0)

                pixel_old = img[r][c]
                
                pixel_new = round(
                    transfer_function[tileindex_r][tileindex_c][pixel_old] * (ratio_r) * (ratio_c) +
                    transfer_function[next_r][tileindex_c][pixel_old] * (1 - ratio_r) * (ratio_c) +
                    transfer_function[tileindex_r][next_c][pixel_old] * (ratio_r) * (1 - ratio_c) +
                    transfer_function[next_r][next_c][pixel_old] * (1 - ratio_r) * (1 - ratio_c)
                )
                img_new[r][c] = pixel_new
        return img_new
    
    # Use local Histogram Equalization: CLAHE
    def CLAHE(image, size):
        # Split image to different grids
        grids = [[0]*size[1] for _ in range(size[0])]
        grid_rowPixel = len(image.img) // size[0]
        grid_colPixel = len(image.img[0]) // size[1]
        for r in range(size[0]):
            for c in range(size[1]):
                startrow, startcol = (r * grid_rowPixel, c * grid_colPixel)
                endrow, endcol = ((r+1) * grid_rowPixel, (c+1) * grid_colPixel)
                row_grid = image.img[startrow:endrow, startcol:endcol]
                grids[r][c] = row_grid

        img_new = Image.applyCLAHE(image.img, grids)
        return Image(img_new)

    # Remove  Noise
    def denoise(image, type, alpha = None, windowsize = None):
        match type:
            case 0: return Denoise.Guassian(image, alpha)
            case 1: return Denoise.Impluse.Impluse_maxmin(image, windowsize)
            case 2: return Denoise.Impluse.Impluse_minmax(image, windowsize)
            case 3: return Denoise.Impluse.Impluse_PMED(image, windowsize)



def plotHistogram(img):
    num = len(img)

    if num == 1:
        plt.hist(img.ravel(), bins=30, edgecolor='black')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram')
    else :
        _, axes = plt.subplots(nrows=1, ncols=num, figsize=(16,8))
        for i in range(num):
            axes[i].hist(img[i][0].ravel(), bins=256, range=(0,255), edgecolor='black')
            axes[i].set_title(img[i][1])
            axes[i].set_xlim([0, 255])

# Flip the original image, then concatenate two images
sample1_img = Image(cv2.imread("hw1_sample_images/sample1.png"))
flipped_img = Image.hflip(sample1_img)
combined_img = Image.concatenate((sample1_img, flipped_img), 0)
cv2.imwrite("result1.png", combined_img.img)

# Convert images to grayscale images
result2_img = Image(cv2.imread("result1.png", cv2.IMREAD_GRAYSCALE))
cv2.imwrite("result2.png", result2_img.img)

# Decrease brightness by dividing the intensity values by 3
sample2_img = Image(cv2.imread("hw1_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE))
result3_img = Image.changeBrightness(sample2_img, 1/3)
cv2.imwrite("result3.png", result3_img.img)

# Increase brightness by multiplying the intensity values by 3
result4_img = Image.changeBrightness(result3_img, 3)
cv2.imwrite("result4.png", result4_img.img)

# Plot the histograms of "sample2.png", "result3.png" and "result4.png"
# plotHistogram([[sample2_img.img, "sample2.png"], [result3_img.img, "result3.png"], [result4_img.img, "result4.png"]])

# Do global histogram equalization
result5_img = Image.histEqualization(sample2_img)
result6_img = Image.histEqualization(result3_img)
result7_img = Image.histEqualization(result4_img)
cv2.imwrite("result5.png", result5_img.img)
cv2.imwrite("result6.png", result6_img.img)
cv2.imwrite("result7.png", result7_img.img)
plotHistogram([[result5_img.img, "result5.png"], [result6_img.img, "result6.png"], [ result7_img.img, "result7.png"]])

# Do local histogram equalization
result8_img = Image.CLAHE(sample2_img, (2, 2))
cv2.imwrite("result8.png", result8_img.img)
plotHistogram([[result5_img.img, "Global Histogram Equalization"], [result8_img.img, "Local Histogram Equalization"]])

# # Adjust the parameters to obtain the most appealing result
sample3_img = Image(cv2.imread("hw1_sample_images/sample3.png", cv2.IMREAD_GRAYSCALE))
result9_img = Image.CLAHE(sample3_img, (10, 10))
cv2.imwrite("result9.png", result9_img.img)

sample4_img = Image(cv2.imread("hw1_sample_images/sample4.png", cv2.IMREAD_GRAYSCALE))

sample5_img = Image(cv2.imread("hw1_sample_images/sample5.png", cv2.IMREAD_GRAYSCALE))
result10_img = Image.denoise(sample5_img, 3, windowsize=7)
result10_img = Image.denoise(result10_img, 3, windowsize=7)
result10_img = Image.denoise(result10_img, 2, windowsize=7)
cv2.imwrite("result10.png", result10_img.img)

sample6_img = Image(cv2.imread("hw1_sample_images/sample6.png", cv2.IMREAD_GRAYSCALE))
result11_img = Image.denoise(sample6_img, 1, windowsize=7)
result11_img = Image.denoise(result11_img, 2, windowsize=7)
cv2.imwrite("result11.png", result11_img.img)

print(f'result10.png PSNR: {Denoise.PSNR(sample4_img, result10_img)}')
print(f'result11.png PSNR: {Denoise.PSNR(sample4_img, result11_img)}')

sample7_img = Image(cv2.imread("hw1_sample_images/sample7.png", cv2.IMREAD_GRAYSCALE))
result12_img = Image.denoise(sample7_img, 1, windowsize=7)
result12_img = Image.denoise(result12_img, 2, windowsize=7)
result12_img = Image.histEqualization(result12_img)
result12_img = Image.denoise(result12_img, 3, windowsize=7)
cv2.imwrite("result12.png", result12_img.img)

print(f'result12.png PSNR: {Denoise.PSNR(sample4_img, result12_img)}')

plt.show()
