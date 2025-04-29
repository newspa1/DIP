import argparse
import cv2
import numpy as np

class Image:
    def getGridBond(grid):
        grid = grid.astype(int)
        bond = 0
        bond += (grid[0, 0] + grid[0, 2] + grid[2, 0] + grid[2, 2]) // 255
        bond += 2 * (grid[0, 1] + grid[1, 0] + grid[1, 2] + grid[2, 1]) // 255
        return bond

    def getBoolGridBond(grid):
        grid = grid.astype(int)
        bond = 0
        bond += (grid[0, 0] + grid[0, 2] + grid[2, 0] + grid[2, 2])
        bond += 2 * (grid[0, 1] + grid[1, 0] + grid[1, 2] + grid[2, 1])
        return bond        

    def getAllVariations(pattern):
        rotations = [np.rot90(pattern, k=i) for i in range(4)]
        mirrors = [np.fliplr(rot) for rot in rotations]
        return rotations + mirrors
    
    def getConditionalPatterns(type):
        conditional_patterns = {}
        if type == "Skeletonize":
            conditional_patterns[4] = np.array(
                Image.getAllVariations([[0, 255, 0], [0, 255, 255], [0, 0, 0]]) +
                Image.getAllVariations([[0, 0, 255], [0, 255, 255], [0, 0, 255]])
            )
            conditional_patterns[6] = np.array(
                Image.getAllVariations([[255, 255, 255], [0, 255, 255], [0, 0, 0]])
            )
            conditional_patterns[7] = np.array(
                Image.getAllVariations([[255, 255, 255], [0, 255, 255], [0, 0, 255]])
            )
            conditional_patterns[8] = np.array(
                Image.getAllVariations([[255, 255, 255], [255, 255, 255], [0, 0, 0]])
            )
            conditional_patterns[9] = np.array(
                Image.getAllVariations([[255, 255, 255], [255, 255, 255], [255, 0, 0]])
            )
            conditional_patterns[10] = np.array(
                Image.getAllVariations([[255, 255, 255], [255, 255, 255], [255, 0, 255]])
            )
            conditional_patterns[11] = np.array(
                Image.getAllVariations([[255, 255, 255], [255, 255, 255], [255, 255, 0]])
            )
        return conditional_patterns

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
    
    def getNeighbors(img, r, c, windowsize):
        size = (img.shape[0], img.shape[1])
        pixels = [img[r][c]]
        for i in range(1, windowsize // 2 + 1):
            pixels.append(img[max(0, r - i)][c])
            pixels.append(img[min(size[0] - 1, r + i)][c])
            pixels.append(img[r][max(0, c - i)])
            pixels.append(img[r][min(size[1] - 1, c + i)])
        return pixels

    def denoise_Impulse(img, windowsize):
        # PMED
        img_new = np.zeros_like(img)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                pixels = Image.getNeighbors(img, r, c, windowsize)
                # img_new[r, c] = Image.minmax(pixels)
                img_new[r, c] = Image.maxmin(pixels)
        
        return img_new

    def connect_gap(img):
        def isWhiteNeighbor(r, c):
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if img[r + dr, c + dc] == 255:
                        return True

        img_new = img.copy()
        for r in range(2, img_new.shape[0] - 2):
            for c in range(2, img_new.shape[1] - 2):
                if img[r, c] == 0 and isWhiteNeighbor(r, c):
                    img_new[r, c] = 255
        
        return img_new

    def flood_fill(img):
        img_new = img.copy()
        stack = [(0, 0)]
        
        while stack:
            r, c = stack.pop()
            if img_new[r, c] != 0:
                continue

            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dr, dc in directions:
                r_new = min(max(r+dr, 0), img_new.shape[0]-1)
                c_new = min(max(c+dc, 0), img_new.shape[1]-1)
                if img_new[r_new, c_new] == 0:
                    stack.append((r_new, c_new))
            
            img_new[r, c] = 255
                        

        return img_new

    def bitNOT(img):
        img_new = np.zeros_like(img)
        for r in range(img_new.shape[0]):
            for c in range(img_new.shape[1]):
                if img[r, c] == 0:
                    img_new[r, c] = 255
                else:
                    img_new[r, c] = 0
        
        return img_new

    def bitOR(img1, img2):
        img_new = np.zeros_like(img1)
        for r in range(img1.shape[0]):
            for c in range(img1.shape[1]):
                if img1[r, c] == 255 or img2[r, c] == 255:
                    img_new[r, c] = 255

        return img_new

    def HoleFilling(img):
        img_floodfill = Image.flood_fill(img)
        # cv2.imwrite("test/result2-1.png", img_floodfill)
        img_invert_floodfill = Image.bitNOT(img_floodfill)
        # cv2.imwrite("test/result2-2.png", img_invert_floodfill)
        img_new = Image.bitOR(img, img_invert_floodfill)
        return img_new

    def MarkImage(img, type):
        conditional_patterns = Image.getConditionalPatterns(type)
        img_mark = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
            
        for r in range(1, img.shape[0]-1):
            for c in range(1, img.shape[1]-1):
                grid = img[r-1:r+2, c-1:c+2]
                bond = Image.getGridBond(grid)
                if bond not in conditional_patterns:
                    continue

                for pattern in conditional_patterns[bond]:
                    if np.array_equal(grid, pattern):
                        img_mark[r, c] = True
                        break

        return img_mark

    def isPatternHit(grid, type):
        bond = Image.getBoolGridBond(grid)
        if type == "Skeletonize":
            if bond == 1:
                return True
            if bond == 2:
                if grid[2, 1] or grid[1, 2] or grid[1, 0] or grid[0, 1]:
                    return True
            if bond == 4:
                if ((grid[0, 1] and grid[1, 2]) or
                    (grid[0, 1] and grid[1, 0]) or
                    (grid[1, 2] and grid[2, 1]) or 
                    (grid[1, 0] and grid[2, 1])):
                    return True
            if bond >= 5:
                if ((grid[0, 0] and grid[0, 1] and grid[1, 0]) or
                    (grid[1, 2] and grid[2, 1] and grid[2, 2])):
                    return True
            if bond >= 6:
                if ((grid[0, 1] and grid[1, 0] and grid[1, 2]) or 
                    (grid[0, 1] and grid[1, 0] and grid[2, 1]) or 
                    (grid[1, 0] and grid[1, 2] and grid[2, 1]) or 
                    (grid[0, 1] and grid[1, 2] and grid[2, 1])):
                    return True

            if 5 <= bond <= 7:
                if ((grid[0, 1] and grid[1, 2] and grid[2, 0] and not grid[0, 2] and not grid[1, 0] and not grid[2, 1]) or 
                    (grid[0, 1] and grid[1, 0] and grid[2, 2] and not grid[0, 0] and not grid[1, 2] and not grid[2, 1]) or 
                    (grid[0, 2] and grid[1, 0] and grid[2, 1] and not grid[0, 1] and not grid[1, 2] and not grid[2, 0]) or 
                    (grid[0, 0] and grid[1, 2] and grid[2, 1] and not grid[0, 1] and not grid[1, 0] and not grid[2, 2])):
                    return True
            
            if bond >= 3:
                if ((grid[0, 0] and grid[0, 2] and (grid[2, 0] or grid[2, 1] or grid[2, 2])) or
                    (grid[0, 0] and grid[2, 0] and (grid[0, 2] or grid[1, 2] or grid[2, 2])) or
                    (grid[2, 0] and grid[2, 2] and (grid[0, 0] or grid[0, 1] or grid[0, 2])) or
                    (grid[0, 2] and grid[2, 2] and (grid[0, 0] or grid[1, 0] or grid[2, 0]))):
                    return True
            
            return False
                
    def EraseImage(img_new, img_mark, type):
        for r in range(1, img_mark.shape[0]-1):
            for c in range(1, img_mark.shape[1]-1):
                if img_mark[r, c]:
                    grid = img_mark[r-1:r+2, c-1:c+2]
                    if not Image.isPatternHit(grid, type):
                        img_new[r, c] = 0
        
        return img_new

    def Skeletonizing(img, times):
        img_new = img.copy()
        for _ in range(times):
            img_mark = Image.MarkImage(img_new, "Skeletonize")
            img_new = Image.EraseImage(img_new, img_mark, "Skeletonize")
    
        return img_new

    def dilation(img, region):
        img_new = np.zeros_like(region, dtype=int)
        shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for r in range(1, img.shape[0]-1):
            for c in range(1, img.shape[1]-1):
                if region[r, c] != 0:
                    for dr, dc in shifts:
                        if img[r + dr, c + dc] == 255:
                            # print(r+dr, c+dc)
                            img_new[r + dr, c + dc] = 255
        
        return img_new

    def Counting(img):
        img_label = np.zeros_like(img, dtype=int)
        label = 1

        for r in range(1, img.shape[0]-1):
            for c in range(1, img.shape[1]-1):
                if img[r, c] == 255 and img_label[r, c] == 0:
                    # Find connected components
                    region = np.zeros_like(img, dtype=int)
                    region[r, c] = label

                    while True:
                        region_new = Image.dilation(img, region)
                        if np.array_equal(region_new, region):
                            break
                        region = region_new
                    img_label[region == 255] = label
                    label += 1
        
        labels_min_max = []
        output_img = np.zeros_like(img_label)
        for i in range(1, label):
            indices = np.where(img_label == i)
            points = list(zip(indices[0], indices[1]))

            r_min = min(point[0] for point in points)
            c_min = min(point[1] for point in points)
            r_max = max(point[0] for point in points)
            c_max = max(point[1] for point in points)
            
            if i == 1:
                for point in points:
                    output_img[point[0], point[1]] = 1
                labels_min_max.append((r_min, c_min, r_max, c_max))
                continue

            isSubset = -1
            for j in range(len(labels_min_max)):
                if (c_min > labels_min_max[j][1] and c_max < labels_min_max[j][3]) or (c_min < labels_min_max[j][1] and c_max > labels_min_max[j][3]):
                    if ((r_max > labels_min_max[j][0]) and (r_min < labels_min_max[j][2])):
                        isSubset = j
                        break
            if isSubset != -1:    
                for point in points:
                    output_img[point[0], point[1]] = isSubset+1
                labels_min_max[j] = (min(r_min, labels_min_max[j][0]), min(c_min, labels_min_max[j][1]), max(r_max, labels_min_max[j][2]), max(c_max, labels_min_max[j][3]))
            else:
                labels_min_max.append((r_min, c_min, r_max, c_max))
                for point in points:
                    output_img[point[0], point[1]] = len(labels_min_max)
            # print(labels_min_max)

        return output_img, labels_min_max

    def Convolution(img, kernel):
        vector = np.zeros_like(img, dtype=float)
        for r in range(1, img.shape[0]-1):
            for c in range(1, img.shape[1]-1):
                energy = 0
                grid = img[r-1:r+2, c-1:c+2]
                mask = grid * kernel
                for i in range(3):
                    for j in range(3):
                        energy += mask[i, j]
                vector[r, c] = energy
        return vector

    def generateTextureImage(texture_vectors, img):
        img_texture = np.zeros((img.shape[0]*3, img.shape[1]*3), dtype=int)
        for i in range(texture_vectors.shape[0]):
            for r in range(texture_vectors[i].shape[0]):
                for c in range(texture_vectors[i].shape[1]):
                    row, col = i // 3, i % 3
                    img_texture[r + row * texture_vectors[i].shape[0], c + col * texture_vectors[i].shape[1]] = texture_vectors[i][r, c]
        cv2.imwrite('laws_texture.png', img_texture)

    def LawMethod(img):
        laws_kernels = np.array(
            [np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 36,
            np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 12,
            np.array([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]]) / 12,
            np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 12,
            np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4,
            np.array([[-1, 2, -1], [0, 0, 0], [1, -2, 1]]) / 4,
            np.array([[-1, -2, -1], [2, 4, 2], [-1, -2, -1]]) / 12,
            np.array([[-1, 0, 1], [2, 0, -2], [-1, 0, 1]]) / 4,
            np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 4]
        )

        texture_vectors = np.zeros((laws_kernels.shape[0], img.shape[0], img.shape[1]), dtype=float)
        for i in range(laws_kernels.shape[0]):
            texture_vectors[i] = Image.Convolution(img, laws_kernels[i])
        
        # Normalize to 255
        for i in range(texture_vectors.shape[0]):
            texture_vectors[i] = texture_vectors[i] / np.max(texture_vectors[i]) * 255

        return texture_vectors
        
    def k_means(vectors_old, k, max_iter=100):
        vectors = np.zeros((vectors_old.shape[1]*vectors_old.shape[2], vectors_old.shape[0]), dtype=int)
        for i in range(vectors_old.shape[0]):
            for r in range(vectors_old.shape[1]):
                for c in range(vectors_old.shape[2]):
                    vectors[r*vectors_old.shape[2]+c, i] = vectors_old[i, r, c]

        centroids = np.random.choice(vectors.shape[0], size=k, replace=False)
        centroids = vectors[centroids]

        converge = False
        clusters = [[] for _ in range(k)]
        labels = np.zeros(vectors.shape[0], dtype=int)
        iter_count = 0
        while not converge:
            clusters = [[] for _ in range(k)]
            for index in range(vectors.shape[0]):
                vector = vectors[index]
                distances_to_centroids = np.zeros(k, dtype=int) 
                for i in range(k):
                    distances_to_centroids[i] = np.linalg.norm(vector - centroids[i])
                cluster_index = np.argmin(distances_to_centroids)
                clusters[cluster_index].append(vector)
                labels[index] = cluster_index
            
            centroids_new = np.zeros((k, 9), dtype=float)
            for i in range(k):
                if clusters[i]:
                    centroids_new[i] = np.mean(clusters[i], axis=0)
                else:
                    centroids_new[i] = centroids[i]
            converge = np.allclose(centroids, centroids_new, atol=1e-2)
            centroids = centroids_new

            iter_count += 1
            if iter_count >= max_iter:
                break
        return labels

    def get_starry_sky(shape):
        patterns = []

        pattern = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        pattern[:] = (5, 5, 20)
        # Add stars
        for _ in range(500):
            y = np.random.randint(0, shape[0])
            x = np.random.randint(0, shape[1])
            brightness = np.random.randint(180, 256)
            pattern[y, x] = (brightness, brightness, brightness)
        patterns.append(pattern)

        x = np.linspace(0, 10, shape[1])
        y = np.linspace(0, 10, shape[0])
        xx, yy = np.meshgrid(x, y)
        noise = (np.sin(xx) * np.cos(yy) + 1) * 127
        pattern = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        pattern[:, :, 2] = noise.astype(np.uint8)       # Red channel (lava)
        pattern[:, :, 1] = (255 - noise).astype(np.uint8) // 4  # slight green (glow)
        patterns.append(pattern)

        cx, cy = shape[0] // 2, shape[1] // 2
        pattern = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

        for y in range(shape[0]):
            for x in range(shape[1]):
                dx = x - cx
                dy = y - cy
                angle = np.arctan2(dy, dx)
                radius = np.sqrt(dx**2 + dy**2)
                swirl = int((angle + radius / 30.0) * 50) % 256
                pattern[y, x] = [(swirl * 2) % 256, swirl, (255 - swirl)]
        patterns.append(pattern)
        return patterns

def problem1_a():
    sample1 = cv2.imread("hw3_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
    result1 = sample1.copy()
    result1 = Image.denoise_Impulse(result1, 5)
    cv2.imwrite('result1.png', result1)
    # cv2.imwrite('test/result1-1.png', result1)

def problem1_b():
    result1 = cv2.imread("result1.png", cv2.IMREAD_GRAYSCALE)
    result2 = Image.HoleFilling(result1)
    cv2.imwrite('result2.png', result2)

def problem1_c():
    result1 = cv2.imread("result1.png", cv2.IMREAD_GRAYSCALE)
    # result3 = result1.copy()

    result3 = Image.Skeletonizing(result1, 50)
    cv2.imwrite('result3.png', result3)

def problem1_d():
    result2 = cv2.imread("result2.png", cv2.IMREAD_GRAYSCALE)
    img_label, label_min_max = Image.Counting(result2)
    img_count = np.zeros((result2.shape[0], result2.shape[1], 3), dtype=np.uint8)
    for i in range(1, len(label_min_max)+1):
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        indices = np.where(img_label == i)
        points = list(zip(indices[0], indices[1]))
        for point in points:
            img_count[point[0], point[1]] = color
    
    print("Number of Object: ", len(label_min_max))
    cv2.imwrite("Count.png", img_count)

def problem2_a():
    sample2 = cv2.imread("hw3_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE)
    texture_vectors = Image.LawMethod(sample2)
    Image.generateTextureImage(texture_vectors, sample2)

def problem2_b():
    sample2 = cv2.imread("hw3_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE)
    texture_vectors = Image.LawMethod(sample2)
    labels = Image.k_means(texture_vectors, 3)
    result4 = np.zeros((sample2.shape[0], sample2.shape[1], 3), dtype=np.uint8)

    for index in range(labels.shape[0]):
        r, c = index // sample2.shape[1], index % sample2.shape[1]
        if labels[index] == 0:
            result4[r, c] = [255, 0, 0]
        if labels[index] == 1:
            result4[r, c] = [0, 255, 0]
        if labels[index] == 2:
            result4[r, c] = [0, 0, 255]

    cv2.imwrite('result4.png', result4)

def problem2_c():
    result4 = cv2.imread("result4.png")
    result5 = np.zeros((result4.shape[0], result4.shape[1], 3), dtype=np.uint8)

    red_mask = np.all(result4 == [0, 0, 255], axis=-1)
    green_mask = np.all(result4 == [0, 255, 0], axis=-1)
    blue_mask = np.all(result4 == [255, 0, 0], axis=-1)

    labels = np.zeros((result4.shape[0], result4.shape[1]), dtype=int)
    labels[red_mask] = 1
    labels[green_mask] = 2
    labels[blue_mask] = 3

    patterns = Image.get_starry_sky((result4.shape[0], result4.shape[1]))

    for r in range(result4.shape[0]):
        for c in range(result4.shape[1]):
            
            if labels[r, c] == 1:
                result5[r, c] = patterns[0][r, c]
            elif labels[r, c] == 2:
                result5[r, c] = patterns[1][r, c]
            else:
                result5[r, c] = patterns[2][r, c]
    
    cv2.imwrite('result5.png', result5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("choice", type=int, choices=range(1, 8))
    args = parser.parse_args()

    options = {
        1: problem1_a,
        2: problem1_b,
        3: problem1_c,
        4: problem1_d,
        5: problem2_a,
        6: problem2_b,
        7: problem2_c,
    }

    options[args.choice]()