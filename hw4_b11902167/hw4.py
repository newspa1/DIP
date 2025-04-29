import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


class Image:
    def apply_dither_matrix(img, dither_matrix):
        dither_size = dither_matrix.shape[0]
        result = np.zeros_like(img)

        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                pixel_value = np.float64(img[r, c])
                threshold = 255 * (dither_matrix[r % dither_size, c % dither_size] + 0.5) / (dither_size**2)
                if pixel_value > threshold:
                    result[r, c] = 255
        
        return result

    def double_dither_matrix(old_dither_matrix):
        old_size = old_dither_matrix.shape[0]
        new_dither_matrix = np.zeros((old_size * 2, old_size * 2), dtype=np.float64)
        
        new_dither_matrix[0:old_size, 0:old_size] = old_dither_matrix * 4 + 1
        new_dither_matrix[0:old_size, old_size:] = old_dither_matrix * 4 + 2
        new_dither_matrix[old_size:, 0:old_size] = old_dither_matrix * 4 + 3
        new_dither_matrix[old_size:, old_size:] = old_dither_matrix * 4 + 0

        return new_dither_matrix

    def apply_diffusion_dithering(img, diffuse_matrix):
        # Normalize to [0, 1]
        float_img = img.astype(np.float64)
        result = np.zeros_like(img)

        for r in range(float_img.shape[0]):
            for c in range(float_img.shape[1]):
                pixel_value = float_img[r, c]
                if pixel_value > 127:
                    result[r, c] = 255
                    error = pixel_value - 255
                else:
                    result[r, c] = 0
                    error = pixel_value

                for dir, weight in diffuse_matrix.items():
                    target_r, target_c = r + dir[0], c + dir[1]
                    if 0 <= target_r < float_img.shape[0] and 0 <= target_c < float_img.shape[1]:
                        float_img[target_r, target_c] += error * weight

        return result

    def remove_horizontal_lines(img):
        for r in range(img.shape[0]):
            # Check horizontal line
            black_ratio = np.sum(img[r, :] == 0) / img.shape[1] 
            if black_ratio > 0.5:
                for c in range(img.shape[1]):
                    if img[r, c] == 0 and (img[max(0, r-1), c] == 255 or img[min(img.shape[0]-1, r+1), c] == 255):
                        img[r, c] = 255
        return img

    def label_components(img):
        labels = np.zeros_like(img, dtype=np.int32)
        label_count = 0
        
        for c in range(img.shape[1]):
            for r in range(img.shape[0]):
                if img[r, c] == 0 and labels[r, c] == 0:
                    label_count += 1
                    labels[r, c] = label_count
                    stack = [(r, c)]

                    while stack:
                        x, y = stack.pop()
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if 0 <= x+dx < img.shape[0] and 0 <= y+dy < img.shape[1] and img[x+dx, y+dy] == 0 and labels[x+dx, y+dy] == 0:
                                    labels[x+dx, y+dy] = label_count
                                    stack.append((x+dx, y+dy))

        return labels, label_count
    
    def shape_verify(img_label, count):
        def containHole(r_max, r_min, c_max, c_min):
            grid = img_label[r_min-1:r_max+2, c_min-1:c_max+2]
            # print(grid)
            stack = [(0, 0)]
            
            while stack:
                x, y = stack.pop()
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if 0 <= x+dx < grid.shape[0] and 0 <= y+dy < grid.shape[1] and grid[x+dx, y+dy] == 0:
                            grid[x+dx, y+dy] = 255
                            stack.append((x+dx, y+dy))

            return np.sum(grid == 0) > 0

        for label in range(1, count + 1):
            indices = np.where(img_label == label)
            
            r_min, r_max = np.min(indices[0]), np.max(indices[0])
            c_min, c_max = np.min(indices[1]), np.max(indices[1])
            height = r_max - r_min
            width = c_max - c_min

            # print(f"Label {label}: Height = {height}, Width = {width}, r_max = {r_max}")

            # Verify the shape
            if height > 70: # Treble Clef
                print(1, end="")
            elif height > 38 and width > 10: 
                print(", ", end="")
                if 22 > width > 10: # Half Note or Quarter Note
                    if containHole(r_max, r_min, c_max, c_min):
                        print(4, end="")
                    else:
                        print(2, end="") 
                elif width > 20: # Eighth Note or Sixteenth Note
                    print(3, end="")

                # Verify pitch
                if r_max > 81:
                    print("(C)", end="")
                elif r_max > 75:
                    print("(D)", end="")
                elif r_max > 70:
                    print("(E)", end="")
                elif r_max > 64:
                    print("(F)", end="")
                elif r_max > 59:
                    print("(G)", end="")
                elif r_max > 53:
                    print("(A)", end="")
                else:
                    print("(B)", end="")

            elif 9 <= width <= 11:
                print("13", end="")
                
    def remove_horizontal_stripes(img):
        f_shift = fftshift(fft2(img))
        center_r, center_c = img.shape[0] // 2, img.shape[1] // 2
        stripe_width = 10

        # Remove horizontal stripes
        mask = np.ones(img.shape, dtype=np.uint8)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                dr, dc = abs(r - center_r), abs(c - center_c)
                if (dc < stripe_width) and (dr > stripe_width // 2 or dc > stripe_width // 2):
                    mask[r, c] = 0.8

        plt.figure(figsize=(10,8))
        plt.title('FFT Magnitude (Log Scale)')
        plt.imshow(np.log(1 + np.abs(f_shift*mask)), cmap='gray')
        plt.colorbar()
        plt.savefig("test/sample4_fft_remove_vertical_line.png")

        img_back = np.abs(ifft2(ifftshift(f_shift*mask)))
        return np.clip(img_back, 0, 255).astype(np.uint8)
    
    def remove_vertical_stripes(img):
        f_shift = fftshift(fft2(img))

        center_r, center_c = img.shape[0] // 2, img.shape[1] // 2
        stripe_width = 10

        # Remove horizontal stripes
        mask = np.ones(img.shape, dtype=np.uint8)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                dr, dc = abs(r - center_r), abs(c - center_c)
                if (dr < stripe_width) and (dr > stripe_width // 2 or dc > stripe_width // 2):
                    mask[r, c] = 0.8

        plt.figure(figsize=(10,8))
        plt.title('FFT Magnitude (Log Scale)')
        plt.imshow(np.log(1 + np.abs(f_shift*mask)), cmap='gray')
        plt.colorbar()
        plt.savefig("test/sample4_fft_remove_horizontal_line.png")

        img_back = np.abs(ifft2(ifftshift(f_shift*mask)))
        return np.clip(img_back, 0, 255).astype(np.uint8)

    def remove_both_stripes(img):
        f_shift = fftshift(fft2(img))

        center_r, center_c = img.shape[0] // 2, img.shape[1] // 2
        stripe_width = 10

        # Remove horizontal stripes
        mask = np.ones(img.shape, dtype=np.uint8)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                dr, dc = abs(r - center_r), abs(c - center_c)
                if (dr < stripe_width or dc < stripe_width) and (dr > stripe_width // 2 or dc > stripe_width // 2):
                    mask[r, c] = 0.8

        plt.figure(figsize=(10,8))
        plt.title('FFT Magnitude (Log Scale)')
        plt.imshow(np.log(1 + np.abs(f_shift*mask)), cmap='gray')
        plt.colorbar()
        plt.savefig("test/sample4_fft_remove_both_lines.png")

        img_back = np.abs(ifft2(ifftshift(f_shift*mask)))
        return np.clip(img_back, 0, 255).astype(np.uint8)

def problem1_a():
    sample1 = cv2.imread("hw4_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
    result1 = sample1.copy()
    dither_matrix = np.array([[1, 2], [3, 0]], dtype=np.float64)

    result1 = Image.apply_dither_matrix(result1, dither_matrix)
    cv2.imwrite("result1.png", result1)

def problem1_b():
    sample1 = cv2.imread("hw4_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
    result2 = sample1.copy()

    dither_matrix = np.array([[1, 2], [3, 0]], dtype=np.float64)
    for _ in range(7):
        dither_matrix = Image.double_dither_matrix(dither_matrix)

    result2 = Image.apply_dither_matrix(result2, dither_matrix / (np.max(dither_matrix) + 1))
    cv2.imwrite("result2.png", result2)

def problem1_c():
    sample1 = cv2.imread("hw4_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)

    # Apply Floyd-Steinberg dithering
    result3 = sample1.copy()
    diffuse_matrix1 = {(0, 1): 7/16, (1, -1): 3/16, (1, 0): 5/16, (1, 1): 1/16}
    result3 = Image.apply_diffusion_dithering(result3, diffuse_matrix1)
    cv2.imwrite("result3.png", result3)

    # Apply Jarvis-Judice-Ninke dithering
    result4 = sample1.copy()
    diffuse_matrix2 = {(0, 1): 7/48, (0, 2): 5/48, (1, -2): 3/48, (1, -1): 5/48, (1, 0): 7/48, (1, 1): 5/48, (1, 2): 3/48, (2, -2): 1/48, (2, -1): 3/48, (2, 0): 5/48, (2, 1): 3/48, (2, 2): 1/48}
    result4 = Image.apply_diffusion_dithering(result4, diffuse_matrix2)
    cv2.imwrite("result4.png", result4)

    # Apply Stucki dithering
    result_tmp = sample1.copy()
    diffuse_matrix3 = {(0, 1): 8/42, (0, 2): 4/42, (1, -2): 2/42, (1, -1): 4/42, (1, 0): 8/42, (1, 1): 4/42, (1, 2): 2/42, (2, -2): 1/42, (2, -1): 2/42, (2, 0): 4/42, (2, 1): 2/42, (2, 2): 1/42}
    result_tmp = Image.apply_diffusion_dithering(result_tmp, diffuse_matrix3)
    cv2.imwrite("test/Stucki_dithering.png", result_tmp)

    # Apply Atkinson dithering
    result_tmp = sample1.copy()
    diffuse_matrix4 = {(0, 1): 1/8, (0, 2): 1/8, (1, -1): 1/8, (1, 0): 1/8, (1, 1): 1/8, (2, 0): 1/8}
    result_tmp = Image.apply_diffusion_dithering(result_tmp, diffuse_matrix4)
    cv2.imwrite("test/Atkinson_dithering.png", result_tmp)

    # Apply Burkes dithering
    result_tmp = sample1.copy()
    diffuse_matrix5 = {(0, 1): 8/32, (0, 2): 4/32, (1, -2): 2/32, (1, -1): 4/32, (1, 0): 8/32, (1, 1): 4/32, (1, 2): 2/32}
    result_tmp = Image.apply_diffusion_dithering(result_tmp, diffuse_matrix5)
    cv2.imwrite("test/Burkes_dithering.png", result_tmp)

def problem2():
    sample2 = cv2.imread("hw4_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE)
    sample3 = cv2.imread("hw4_sample_images/sample3.png", cv2.IMREAD_GRAYSCALE)

    # Sample2
    sample2 = np.where(sample2 > 180, 255, 0).astype(np.uint8)
    sample2 = Image.remove_horizontal_lines(sample2)
    sample2_label, count = Image.label_components(sample2)

    print("Sample2: ", end="")
    Image.shape_verify(sample2_label, count)
    cv2.imwrite("test/sample2.png", sample2)
    print()

    # Sample3
    sample3 = np.where(sample3 > 180, 255, 0).astype(np.uint8)
    sample3 = Image.remove_horizontal_lines(sample3)
    sample3_label, count = Image.label_components(sample3)

    print("Sample3: ", end="")
    Image.shape_verify(sample3_label, count)
    cv2.imwrite("test/sample3.png", sample3)

def problem3():
    sample4 = cv2.imread("hw4_sample_images/sample4.png", cv2.IMREAD_GRAYSCALE)
    result5 = Image.remove_horizontal_stripes(sample4)
    result6 = Image.remove_vertical_stripes(sample4)
    result7 = Image.remove_both_stripes(sample4)
    cv2.imwrite("result5.png", result5)
    cv2.imwrite("result6.png", result6)
    cv2.imwrite("result7.png", result7)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("choice", type=int, choices=range(1, 8))
    args = parser.parse_args()

    options = {
        1: problem1_a,
        2: problem1_b,
        3: problem1_c,
        4: problem2,
        5: problem3
    }

    options[args.choice]()