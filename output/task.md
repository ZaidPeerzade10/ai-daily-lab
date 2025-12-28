# AI Daily Lab — 2025-12-28

## Task
1. Generate a 2D NumPy array (e.g., 64x64 pixels) initialized with zeros, representing a black grayscale image.
2. Add several synthetic 'objects' to this image using NumPy slicing and broadcasting:
    *   A bright square in the center (e.g., value 200).
    *   A horizontal bright line (e.g., value 150).
    *   A diagonal bright line (e.g., value 100) from top-left to bottom-right.
3. Implement a basic 2D convolution function `manual_convolve2d(image, kernel)` using NumPy operations (avoid `scipy.signal.convolve2d` for this task). Your function should handle padding or border effects as you deem appropriate (e.g., 'same' padding).
4. Define a 3x3 'edge detection' kernel (e.g., a simple Sobel or Laplacian approximation).
5. Apply your `manual_convolve2d` function with the edge detection kernel to your generated image.
6. Visualize the original synthetic image and the convolved (edge-detected) image side-by-side using `matplotlib.pyplot.imshow`. Use a 'gray' colormap and ensure both plots have appropriate titles (e.g., 'Original Image', 'Edge Detected Image').

## Focus
numpy, data visualization, basic AI experimentation

## Dataset
Synthetic 2D NumPy array (image-like data)

## Hint
For adding objects, utilize direct indexing and assignment like `img[row_slice, col_slice] = value`. For manual convolution, consider creating a padded version of the image to simplify boundary handling. The convolution operation involves iterating through the image, taking element-wise products with the kernel, and summing them. `matplotlib.pyplot.imshow(array, cmap='gray')` is your friend for visualization.
