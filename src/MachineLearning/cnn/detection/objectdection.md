### classification with localization

$$
\begin{aligned}

& \text{mid point} : b_x,b_y \\
& \text{height and wigth} : b_w, b_y \\

& \text{target label y} :
\begin{bmatrix}
p_c : \text{is there an object (1 or 0)}\\
b_x \\
b_y \\
b_w \\
b_y \\
c_1 \\
\vdots\\
c_n :\text{n class}\\

\end{bmatrix}

\end{aligned}
$$


### landmark detection
...

## object detection

### sliding windows detection

convolution implementation of sliding windows

weakness : position of bounding box is not going to be too accurate

### yolo algorithm

divide the image into n grid cell,
for each grid cell target label y equal to classification with localization target label y
so the target out size $n \times n \times y_n$

### IOU(intersection over union) 交并比

### Non-max suppression example