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
                img_new[r, c] = Image.minmax(pixels)
        
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
        cv2.imwrite("test/result2-1.png", img_floodfill)
        img_invert_floodfill = Image.bitNOT(img_floodfill)
        cv2.imwrite("test/result2-2.png", img_invert_floodfill)
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
                        # print(r, c)
                        # print("===")
                        region_new = Image.dilation(img, region)
                        if np.array_equal(region_new, region):
                            break
                        region = region_new
                    # print("================================")
                    # for rr in range(1, region.shape[0]-1):
                    #     for cc in range(1, region.shape[1]-1):
                    #         if region[rr, cc] == 255:
                    #             print(rr, cc)
                    img_label[region == 255] = label
                    label += 1
                # print("label", label)
        
        return label - 1

def problem1_a():
    sample1 = cv2.imread("hw3_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
    result1 = sample1.copy()
    result1 = Image.denoise_Impulse(result1, 5)
    cv2.imwrite('result1.png', result1)

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
    result4 = result2.copy()

    result4 = Image.Counting(result4)
    print("Number of Object: ", result4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("choice", type=int, choices=range(1, 8))
    args = parser.parse_args()

    options = {
        1: problem1_a,
        2: problem1_b,
        3: problem1_c,
        4: problem1_d,
    }

    options[args.choice]()