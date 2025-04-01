# DIP Homework Assignment #2

## Problem 1: MORPHOLOGICAL PROCESSING

### (a)


### (b)
<div style="display: flex; justify-content: center; gap: 20px;">
  <figure>
    <img src="result1.png" width="300">
    <figcaption style=" text-align: center">result1.png</figcaption>
  </figure>
  <figure>
    <img src="result2.png" width="300">
    <figcaption style=" text-align: center">result2.png</figcaption>
  </figure>
</div>

#### Approach
1. Fill the background with white by using **dfs**.
<div style="display: flex; justify-content: center; gap: 20px;">
  <figure>
    <img src="test/result2-1.png" width="300">
    <figcaption style=" text-align: center">result2-1.png</figcaption>
  </figure>
</div>

2. Inverte the image to `result2-1.png` (perform bitNOT).
<div style="display: flex; justify-content: center; gap: 20px;">
  <figure>
    <img src="test/result2-2.png" width="300">
    <figcaption style=" text-align: center">result2-1.png</figcaption>
  </figure>
</div>

3. Perform bitOR to `result2.png` and `result2-2.png`, Obtaining the image with hole filling.

#### Discussion