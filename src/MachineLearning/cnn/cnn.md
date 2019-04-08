# CNN (convvolution neural networks)

## convolution
$$
\begin{aligned}
& \text{image : } n \times n\\
& \text{filter(kernl) : } f \times f\\
& \text{padding: } p \\
& \text{stride : } s\\
& \text{output : } [\frac{n+2p-f}{s}+1] \times [\frac{n+2p-f}{s}+1]
\end{aligned}
$$

## convolution layer l
$$
\begin{aligned}
& \text{input : }\quad n^{[l-1]} \times n^{[l-1]} \times n^{[l-1]}_{c}\\\\

& \text{filter(kernl) size : }\quad f^{[l]}\\
& \text{padding: }\quad p^{[l]} \\
& \text{stride : }\quad s^{[l]}\\
& \text{numbers of filter : }\quad n^{[l]}_{c}\\\\

& \text{each filter : }\quad f^{[l]} \times f^{[l]} \times n^{[l-1]}_{c}\\
& \text{activations: }\quad a^{[l]} = n^{[l]} \times n^{[l]} \times n^{[l]}_{c} \\
& \text{weights : }\quad f^{[l]} \times f^{[l]} \times n^{[l-1]}_{c} \times n^{[l]}_{c}\\
& \text{bias : }\quad  1 \times 1 \times 1 \times n^{[l]}_{c} \\\\

& \text{output : }\quad n^{[l]} \times n^{[l]} \times n^{[l]}_{c}\\
& n^{[l]} = [\frac{n^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1]
\end{aligned}
$$

---

## pooling layer

$$
\begin{aligned}
& \text{input:} \quad n \times n \times n_{c}\\

&\text{filter:} \quad f\\
&\text{stride:} \quad s\\
&\text{MAx or Average pooling}\\

&\text{output:} \quad [\frac{n-f}{s}+1] \times [\frac{n-f}{s}+1]  \times n_{c}\\

\end{aligned}
$$

## fully connected layer

## why convolution

1. Parameter sharing : a feature decetor(filter) that's useful in one part of the image is probably useful in another part of image
2. Sparsity of connections : in each layer, each output value depends only on a small number of inputs