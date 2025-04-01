import argparse
import cv2
import numpy as np

class Image:
    def getGridBond(grid):
        grid = grid.astype(int)
        bond = 0
        bond += (grid[0, 0] + grid[0, 2] + grid[2, 0] + grid[2, 2]) // 255
        bond += 2* (grid[0, 1] + grid[1, 0] + grid[1, 2] + grid[2, 1]) // 255
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

    def denoise_Guassian(img, kernelsize):
        kernel = np.ones((kernelsize, kernelsize), np.float32) / (kernelsize**2)
        img_new = img.copy()
        for r in range(kernelsize//2, img_new.shape[0] - kernelsize//2):
            for c in range(kernelsize//2, img_new.shape[1] - kernelsize//2):
                sub_img = img_new[r-kernelsize//2:r+kernelsize//2+1, c-kernelsize//2:c+kernelsize//2+1]
                img_new[r, c] = np.sum(sub_img * kernel)

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
        if type == "Skeletonize":
            patterns = np.array(
                Image.getAllVariations([[False, False, False], [False, True, False], [False, False, True]]) +
                Image.getAllVariations([[False, False, False], [False, True, False], [False, True, False]]) + 
                Image.getAllVariations([[False, False, False], [False, True, True], [False, True, False]])
            )
            for pattern in patterns:
                if np.array_equal(grid, pattern):
                    return True

            patterns = np.array(
                np.array([[True, True, False], [True, True, False], [False, False, False]]) +
                np.array([[False, False, False], [False, True, True], [False, True, True]]) +
                Image.getAllVariations([[False, True, False], [True, True, True], [False, False, False]]),
                dtype=bool
            )
            for pattern in patterns:
                isfind = True
                for r in range(3):
                    for c in range(3):
                        if pattern[r, c] and not grid[r, c]:
                            isfind = False
                
                if isfind:
                    return True
                
            # print("second")
            patterns = np.array(
                Image.getAllVariations([[1, 2, 1], [2, 1, 2], [3, 3, 3]])
            )
            for pattern in patterns:
                isfind = True
                true1 = False
                for r in range(3):
                    for c in range(3):
                        if pattern[r, c] == 1 and not grid[r, c]:
                            isfind = False
                        if pattern[r, c] == 3 and grid[r, c]:
                            true1 = True

                if isfind and true1:
                    return True
            
            patterns = np.array(
                Image.getAllVariations([[2, 1, 0], [0, 1, 1], [1, 0, 2]])
            )

            for pattern in patterns:
                isfind = True
                for r in range(3):
                    for c in range(3):
                        if pattern[r, c] == 1 and not grid[r, c]:
                            isfind = False
                        if pattern[r, c] == 0 and grid[r, c]:
                            isfind = False

                if isfind:
                    return True

            return False

    def EraseImage(img_new, img_mark, type):
        # unconditional_patterns = Image.getUnconditionalPatterns(type)
        for r in range(1, img_mark.shape[0]-1):
            for c in range(1, img_mark.shape[1]-1):
                grid = img_mark[r-1:r+2, c-1:c+2]
                # print(Image.isPatternHit(grid, type))
                if Image.isPatternHit(grid, type):
                    img_new[r, c] = 0
        
        return img_new

    def Skeletonizing(img, times):
        img_new = img.copy()
        for _ in range(times):
            img_mark = Image.MarkImage(img_new, "Skeletonize")
            img_new = Image.EraseImage(img_new, img_mark, "Skeletonize")
        
        return img_new

def problem1_a():
    sample1 = cv2.imread("hw3_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
    result1 = sample1.copy()
    result1 = Image.denoise_Guassian(result1, 3)
    threshold = 150
    result1[result1 > threshold] = 255
    result1[result1 <= threshold] = 0
    
    result1 = Image.denoise_Guassian(result1, 3)
    threshold = 150
    result1[result1 > threshold] = 255
    result1[result1 <= threshold] = 0

    cv2.imwrite('result1.png', result1)

def problem1_b():
    result1 = cv2.imread("result1.png", cv2.IMREAD_GRAYSCALE)
    result2 = Image.HoleFilling(result1)
    cv2.imwrite('result2.png', result2)

def problem1_c():
    result1 = cv2.imread("result1.png", cv2.IMREAD_GRAYSCALE)
    # result3 = result1.copy()

    result3 = Image.Skeletonizing(result1, 20)
    cv2.imwrite('result3.png', result3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("choice", type=int, choices=range(1, 8))
    args = parser.parse_args()

    options = {
        1: problem1_a,
        2: problem1_b,
        3: problem1_c,
    }

    options[args.choice]()