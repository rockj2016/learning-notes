# [雅可比矩阵](https://zh.wikipedia.org/wiki/%E9%9B%85%E5%8F%AF%E6%AF%94%E7%9F%A9%E9%98%B5)
$$
\text{给定向量}
\bf{x} = 
\begin{bmatrix}
 x{_1} \\ x{_2} \\ \vdots \\ x{_n}
\end{bmatrix}
,
\bf{y} = 
\begin{bmatrix}
 y{_1} \\ y{_2} \\ \vdots \\ y{_m}
\end{bmatrix}
$$
设函数f是一个从n维欧式空间映射到m维欧式空间的函数,函数由m个实数组成:
$$
y{_1}(x{_1},\cdots,x{_n}),\cdots,y{_m}(x{_1},\cdots,x{_n})
$$
则这些函数的偏导数可以组成一个m x n的矩阵,这个矩阵就是雅可比矩阵
$$
\begin{bmatrix}
\frac{\partial{y{_1}}}{\partial{x{_1}}} & \cdots & \frac{\partial{y{_1}}}{\partial{x{_n}}} \\
\vdots & \ddots & \vdots \\
\frac{\partial{y{_m}}}{\partial{x{_1}}} & \cdots & \frac{\partial{y{_m}}}{\partial{x{_n}}} \\
\end{bmatrix}
$$

# [矩阵求导布局规范](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions)

| matrix | vector | scalar |
| ------------- | ------------- | ------------- |
| $\bf{X},\bf{Y}$ | $\bf{x},\bf{y}$ | $x,y$ |

本质上的问题是:当向量对向量求导时,即$\frac{\partial{\bf{y}}}{\partial{\bf{x}}}$,可以写作成两种矛盾的格式.假设$\bf{y}$是m维列向量,$\bf{x}$是n维度列向量,则求导结果可以是 n×m matrix 也可以是m×n matrix

- Numerator layout: 求导结果根据$\bf{y}$和$\bf{x{^T}}$布局,这也就是Jacobian formulation.$\frac{\partial{\bf{y}}}{\partial{x}}$布局为行向量,$\frac{\partial{y}}{\partial{\bf{x}}}$布局为列向量

- Denominator layout : 也被叫做Hessian formulation,是Jacobian formulation的转置
<table class="wikitable">
<caption>Result of differentiating various kinds of aggregates with other kinds of aggregates
</caption>
<tbody><tr>
<th colspan="2" rowspan="2">
</th>
<th colspan="2">Scalar <i>y</i>
</th>
<th colspan="2">Column vector <b>y</b> (size <i>m</i>×<i>1</i>)
</th>
<th colspan="2">Matrix <b>Y</b> (size <i>m</i>×<i>n</i>)
</th></tr>
<tr>
<th>Notation</th>
<th>Type
</th>
<th>Notation</th>
<th>Type
</th>
<th>Notation</th>
<th>Type
</th></tr>
<tr>
<th rowspan="2">Scalar <i>x</i>
</th>
<th>Numerator
</th>
<td rowspan="2" style="text-align:center;"><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\frac {\partial y}{\partial x}}}">
  <semantics>
    <mrow class="MJX-TeXAtom-ORD">
      <mstyle displaystyle="true" scriptlevel="0">
        <mrow class="MJX-TeXAtom-ORD">
          <mfrac>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mi>y</mi>
            </mrow>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mi>x</mi>
            </mrow>
          </mfrac>
        </mrow>
      </mstyle>
    </mrow>
    <annotation encoding="application/x-tex">{\displaystyle {\frac {\partial y}{\partial x}}}</annotation>
  </semantics>
</math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/0deac2b96aa5d0329450647f183f9365584c67b2" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.005ex; width:3.484ex; height:5.676ex;" alt="\frac{\partial y}{\partial x}"></span>
</td>
<td rowspan="2">Scalar
</td>
<td rowspan="2" style="text-align:center;"><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\frac {\partial \mathbf {y} }{\partial x}}}">
  <semantics>
    <mrow class="MJX-TeXAtom-ORD">
      <mstyle displaystyle="true" scriptlevel="0">
        <mrow class="MJX-TeXAtom-ORD">
          <mfrac>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi mathvariant="bold">y</mi>
              </mrow>
            </mrow>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mi>x</mi>
            </mrow>
          </mfrac>
        </mrow>
      </mstyle>
    </mrow>
    <annotation encoding="application/x-tex">{\displaystyle {\frac {\partial \mathbf {y} }{\partial x}}}</annotation>
  </semantics>
</math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/67d5f2cf89374e95eb31cdf816533244b4d45d1d" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.005ex; width:3.565ex; height:5.676ex;" alt="\frac{\partial \mathbf{y}}{\partial x}"></span>
</td>
<td>Size-<i>m</i> <a href="/wiki/Column_vector" class="mw-redirect" title="Column vector">column vector</a>
</td>
<td rowspan="2" style="text-align:center;"><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\frac {\partial \mathbf {Y} }{\partial x}}}">
  <semantics>
    <mrow class="MJX-TeXAtom-ORD">
      <mstyle displaystyle="true" scriptlevel="0">
        <mrow class="MJX-TeXAtom-ORD">
          <mfrac>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi mathvariant="bold">Y</mi>
              </mrow>
            </mrow>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mi>x</mi>
            </mrow>
          </mfrac>
        </mrow>
      </mstyle>
    </mrow>
    <annotation encoding="application/x-tex">{\displaystyle {\frac {\partial \mathbf {Y} }{\partial x}}}</annotation>
  </semantics>
</math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/565884c84274a792e9b5af680a30f550eaf5e3a6" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.005ex; width:4.174ex; height:5.509ex;" alt="\frac{\partial \mathbf{Y}}{\partial x}"></span>
</td>
<td><i>m</i>×<i>n</i> matrix
</td></tr>
<tr>
<th>Denominator
</th>
<td>Size-<i>m</i> <a href="/wiki/Row_vector" class="mw-redirect" title="Row vector">row vector</a>
</td>
<td>
</td></tr>
<tr>
<th rowspan="2">Column vector <b>x</b><br>(size <i>n</i>×<i>1</i>)
</th>
<th>Numerator
</th>
<td rowspan="2" style="text-align:center;"><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\frac {\partial y}{\partial \mathbf {x} }}}">
  <semantics>
    <mrow class="MJX-TeXAtom-ORD">
      <mstyle displaystyle="true" scriptlevel="0">
        <mrow class="MJX-TeXAtom-ORD">
          <mfrac>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mi>y</mi>
            </mrow>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi mathvariant="bold">x</mi>
              </mrow>
            </mrow>
          </mfrac>
        </mrow>
      </mstyle>
    </mrow>
    <annotation encoding="application/x-tex">{\displaystyle {\frac {\partial y}{\partial \mathbf {x} }}}</annotation>
  </semantics>
</math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/01a7fae63303065a57b24c2bb67ab80468a24263" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.005ex; width:3.565ex; height:5.676ex;" alt="\frac{\partial y}{\partial \mathbf{x}}"></span>
</td>
<td>Size-<i>n</i> <a href="/wiki/Row_vector" class="mw-redirect" title="Row vector">row vector</a>
</td>
<td rowspan="2" style="text-align:center;"><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\frac {\partial \mathbf {y} }{\partial \mathbf {x} }}}">
  <semantics>
    <mrow class="MJX-TeXAtom-ORD">
      <mstyle displaystyle="true" scriptlevel="0">
        <mrow class="MJX-TeXAtom-ORD">
          <mfrac>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi mathvariant="bold">y</mi>
              </mrow>
            </mrow>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi mathvariant="bold">x</mi>
              </mrow>
            </mrow>
          </mfrac>
        </mrow>
      </mstyle>
    </mrow>
    <annotation encoding="application/x-tex">{\displaystyle {\frac {\partial \mathbf {y} }{\partial \mathbf {x} }}}</annotation>
  </semantics>
</math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/734fea892fc38deec1d53fa88abed4ca213c0d25" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.005ex; width:3.565ex; height:5.676ex;" alt="\frac{\partial \mathbf{y}}{\partial \mathbf{x}}"></span>
</td>
<td><i>m</i>×<i>n</i> matrix
</td>
<td rowspan="2" style="text-align:center;"><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\frac {\partial \mathbf {Y} }{\partial \mathbf {x} }}}">
  <semantics>
    <mrow class="MJX-TeXAtom-ORD">
      <mstyle displaystyle="true" scriptlevel="0">
        <mrow class="MJX-TeXAtom-ORD">
          <mfrac>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi mathvariant="bold">Y</mi>
              </mrow>
            </mrow>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi mathvariant="bold">x</mi>
              </mrow>
            </mrow>
          </mfrac>
        </mrow>
      </mstyle>
    </mrow>
    <annotation encoding="application/x-tex">{\displaystyle {\frac {\partial \mathbf {Y} }{\partial \mathbf {x} }}}</annotation>
  </semantics>
</math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/433f2d2da465f5a3f2aa8dff5c9d6dd8e9947eef" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.005ex; width:4.174ex; height:5.509ex;" alt="\frac{\partial \mathbf{Y}}{\partial \mathbf{x}}"></span>
</td>
<td rowspan="2">
</td></tr>
<tr>
<th>Denominator
</th>
<td>Size-<i>n</i> <a href="/wiki/Column_vector" class="mw-redirect" title="Column vector">column vector</a>
</td>
<td><i>n</i>×<i>m</i> matrix
</td></tr>
<tr>
<th rowspan="2">Matrix <b>X</b><br>(size <i>p</i>×<i>q</i>)
</th>
<th>Numerator
</th>
<td rowspan="2" style="text-align:center;"><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\frac {\partial y}{\partial \mathbf {X} }}}">
  <semantics>
    <mrow class="MJX-TeXAtom-ORD">
      <mstyle displaystyle="true" scriptlevel="0">
        <mrow class="MJX-TeXAtom-ORD">
          <mfrac>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mi>y</mi>
            </mrow>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi mathvariant="bold">X</mi>
              </mrow>
            </mrow>
          </mfrac>
        </mrow>
      </mstyle>
    </mrow>
    <annotation encoding="application/x-tex">{\displaystyle {\frac {\partial y}{\partial \mathbf {X} }}}</annotation>
  </semantics>
</math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/877eb58a8159dedbc4bc47afc9749803d75d5e35" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.005ex; width:4.174ex; height:5.676ex;" alt="\frac{\partial y}{\partial \mathbf{X}}"></span>
</td>
<td><i>q</i>×<i>p</i> matrix
</td>
<td rowspan="2" style="text-align:center;"><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\frac {\partial \mathbf {y} }{\partial \mathbf {X} }}}">
  <semantics>
    <mrow class="MJX-TeXAtom-ORD">
      <mstyle displaystyle="true" scriptlevel="0">
        <mrow class="MJX-TeXAtom-ORD">
          <mfrac>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi mathvariant="bold">y</mi>
              </mrow>
            </mrow>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi mathvariant="bold">X</mi>
              </mrow>
            </mrow>
          </mfrac>
        </mrow>
      </mstyle>
    </mrow>
    <annotation encoding="application/x-tex">{\displaystyle {\frac {\partial \mathbf {y} }{\partial \mathbf {X} }}}</annotation>
  </semantics>
</math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/86a7d5bedcc1bc202bd55040b26137a6c1740850" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.005ex; width:4.174ex; height:5.676ex;" alt="\frac{\partial \mathbf{y}}{\partial \mathbf{X}}"></span>
</td>
<td rowspan="2">
</td>
<td rowspan="2" style="text-align:center;"><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML" alttext="{\displaystyle {\frac {\partial \mathbf {Y} }{\partial \mathbf {X} }}}">
  <semantics>
    <mrow class="MJX-TeXAtom-ORD">
      <mstyle displaystyle="true" scriptlevel="0">
        <mrow class="MJX-TeXAtom-ORD">
          <mfrac>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi mathvariant="bold">Y</mi>
              </mrow>
            </mrow>
            <mrow>
              <mi mathvariant="normal">∂<!-- ∂ --></mi>
              <mrow class="MJX-TeXAtom-ORD">
                <mi mathvariant="bold">X</mi>
              </mrow>
            </mrow>
          </mfrac>
        </mrow>
      </mstyle>
    </mrow>
    <annotation encoding="application/x-tex">{\displaystyle {\frac {\partial \mathbf {Y} }{\partial \mathbf {X} }}}</annotation>
  </semantics>
</math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/0d7d1744e8920b3885bde9168c70643df3a49cd3" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -2.005ex; width:4.174ex; height:5.509ex;" alt="\frac{\partial \mathbf{Y}}{\partial \mathbf{X}}"></span>
</td>
<td rowspan="2">
</td></tr>
<tr>
<th>Denominator
</th>
<td><i>p</i>×<i>q</i> matrix
</td></tr></tbody></table>


# neural network

## [activation functions](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)

$$
singmod = ...
$$

$$
tanh(x) = \frac{e{^z}-e{^{-z}}}{e{^z}+e{^{-z}}}
$$

$$
\text{rectified linear unit:   }relu(z) = 
\begin{cases} 
0, & \text {if $z<0$} \\ 
z, & \text{if $z>0$} 
\end{cases} 
$$

$$
\text{leaking relu}(z) = 
\begin{cases} 
0.0 1z, & \text {if $z<0$} \\ 
z, & \text{if $z>0$} 
\end{cases} 
$$

$$
\text{softmax: }  \hat{y_{i}} =p_{i} =  
\frac{e^{z_{i}}}{\sum_{k=1}^{m}e^{z_{k}}}
$$
---
## [loss function](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)

$$
\text{Cross Entropy:  }
E = 
\begin{cases}
-(y\log{(p)} + (1-y)\log{(1-p)}) & 
\text{binary classification}\\
-\sum_{i=1}^{m}y_{i}\log{(p_{i})} & 
\text{multiclass classification,  m is the number of classes }
    
\end{cases}
$$
---
## Derivative of Cross Entropy and Softmax
$
\bf{y}\in{\bf{R}^{m\times1}},
\bf{p}\in{\bf{R}^{m\times1}}
$对loss function 求导:
$$
\frac{\partial{E}}{\partial{p_{j}}} = 
\frac{-\sum_{i=1}^{m}y_{i}\log{(p_{i})}}{\partial{p_{j}}} = 
-\frac{y_{j}}{p_{j}}
$$

$$
\frac{\partial{E}}{\partial{\bf{P}}} = 
\begin{bmatrix}
-\frac{y_{1}}{p_{1}}　& \cdots & -\frac{y_{m}}{p_{m}}
\end{bmatrix}=
(-\frac{\bf{y}}{\bf{p}})^{T}
, \in{\bf{R}^{1\times m}}
$$

对 activation functions 求导
$$
\frac{\partial{p_{i}}}{\partial{z_{j}}} = \frac{\frac{e^{z_{i}}}{\sum_{k=1}^{m}e^{z_{k}}}}{z_{j}}
= 
\begin{cases}
p_{i}(1-p_{i})& j=i \\
-p_{i}p_{j}& j\neq{i}
\end{cases}
, \in{\bf{R}^{m\times m}}
$$

$$
\frac{\partial{P}}{\partial{\bf{Z}}} \in \bf{R}^{m \times m}
$$

综合以上

$$
\begin{bmatrix}
-\frac{y_{1}}{p_{1}}　& \cdots & -\frac{y_{m}}{p_{m}}
\end{bmatrix}
\begin{bmatrix}
p_{1}(1-p_{1}) & -p_{1}p_{2} & \cdots & -p_{1}p_{m} \\
-p_{2}p_{1} & p_{2}(1-p_{2}) &  \cdots & -p_{2}p_{m} \\
\vdots & \vdots & \ddots & \vdots\\
-p_{m}p_{1} & -p_{m}p_{2} &  \cdots & p_{m}(1-p_{m}) \\ 
\end{bmatrix}=
\begin{bmatrix}
p_{1}-y_{1} &
p_{2}-y_{2} &
\cdots &
p_{m}-y_{m}&
\end{bmatrix}

$$

$$
\frac{\partial{E}}{\partial{Z}} = \frac{\partial{E}}{\partial{\bf{P}}} \cdot
\frac{\partial{P}}{\partial{\bf{Z}}} = 
(p-y)^{T}
,\in{\bf{R^{1 \times m}}}
$$

若拆开计算,特别注意

$$
\frac{\partial{E}}{\partial{z_{j}}} = \sum_{i = 1}^{m}
\frac{\partial{E}}{\partial{p_{i}}} 
\frac{\partial{p_{i}}}{\partial{z_{j}}}
$$
---

## 对于单层神经网络

输入层$\bf{x}\in{R^{a \times 1}}$,
隐藏层$\bf{h}\in{R^{b \times 1}}$,
输出层$\bf{o}\in{R^{c \times 1}}$,

$$
\begin{aligned}
& \bf{z}^{[1]} = \bf{W}^{[1]}\bf{x} + \bf{b}^{1} \\
& \bf{h} = sigmoid(\bf{z}^{[1]}) \\
& \bf{z}^{[2]} = \bf{W}^{[2]}\bf{h} + \bf{b}^{[2]} \\
& \bf{p} = softmax(\bf{z^{[2]}}) \\
& \bf{E} = -\bf{y}^{T}\log{(\bf{p})}\\
\end{aligned}
$$

需要求
$\frac{\partial{E}}{\partial{w}^{[1]}}$,
$\frac{\partial{E}}{\partial{b}^{[1]}}$,
$\frac{\partial{E}}{\partial{h}}$
$\frac{\partial{E}}{\partial{w}^{[2]}}$,
$\frac{\partial{E}}{\partial{b}^{[2]}}$

$$
\begin{aligned}

&dz^{[2]} = \frac{\partial{E}}{\partial{\bf{z}}^{[2]}} = (p-y)^{T} \\\\

& \frac{\partial{E}}{\partial{\bf{W}}^{[2]}} = dz^{[2]} \frac{{\partial{\bf{z}^{[2]}}}}{{\partial{\bf{W}^{[2]}}}} \\\\

& \frac{{\partial{\bf{z}^{[2]}}}}{{\partial{\bf{W}^{[2]}}}} = 
\begin{bmatrix}
\frac{{\partial{\bf{z}^{[2]}_{1}}}}{{\partial{\bf{W}^{[2]}}}} \\\\
\frac{{\partial{\bf{z}^{[2]}_{2}}}}{{\partial{\bf{W}^{[2]}}}} \\\\
\vdots \\\\ 
\frac{{\partial{\bf{z}^{[2]}_{c}}}}{{\partial{\bf{W}^{[2]}}}}
\end{bmatrix} \in{R^{c \times 1}}\\\\\\

&\frac{{\partial{\bf{z}^{[2]}_{i}}}}{{\partial{{W}_{xy}^{[2]}}}}=
\frac{\partial{\sum_{k=1}^{b}w_{ik}h_{k}}}{{\partial{W_{xy}^{[2]}}}}\\\\

& 可知当 x=i,y=j时\frac{{\partial{\bf{z}^{[2]}_{i}}}}{{\partial{{W}_{xy}^{[2]}}}} = h_{y} \\\\

& \frac{{\partial{\bf{z}^{[2]}_{i}}}}{{\partial{\bf{W}^{[2]}}}} = 
\begin{bmatrix}
0&0&\cdots&0\\
\vdots&\vdots& \ddots &\vdots\\
h_{1} &h_{2} &\cdots & h_{b}\\
\vdots&\vdots& \ddots &\vdots\\
0&0&\cdots&0\\
\end{bmatrix} \in{R^{c \times b}}\\\\

& 所以有\frac{\partial{E}}{\partial{\bf{W}}^{[2]}} = 
\sum_{i=1}^{c}
\frac{{\partial{\bf{z}^{[2]}_{i}}}}{{\partial{\bf{W}^{[2]}}}}
(p_{i}-y_{i}) = 
\begin{bmatrix}
(p_{1}-y_{1})h_{1} & (p_{1}-y_{1})h_{2}& \cdots &(p_{1}-y_{1})h_{b} \\\\
(p_{2}-y_{2})h_{1} & (p_{2}-y_{2})h_{2}& \cdots &(p_{2}-y_{2})h_{b} \\\\
\vdots & \vdots& \ddots &\vdots \\\\
(p_{c}-y_{c})h_{1} & (p_{c}-y_{c})h_{2}& \cdots &(p_{2}-y_{2})h_{b} \\\\
\end{bmatrix} \in{R^{c \times b}}\\\\

&即 : \frac{\partial{E}}{\partial{\bf{W}}^{[2]}} = d{z^{[2]}}^{T}h^{T}\\\\

&\frac{\partial{E}}{\partial{\bf{b}}^{[2]}} = d{z^{[2]}}^{T}\\\\

& d{z^{[1]}}=\frac{\partial{E}}{\partial{h}}\frac{\partial{h}}{\partial{z^{[1]}}} = d{z^{[2]}}
\frac{\partial{\bf{z}^{[2]}}}{\partial{h}}\frac{\partial{h}}{\partial{z^{[1]}}}
=  d{z^{[2]}} \bf{W^{[2]}} \circ sigmoid'(z^{[1]}) \\\\

& \frac{\partial{E}}{\partial{b}^{[1]}} = \frac{\partial{E}}{\partial{h}} 
\frac{\partial{h}}{\partial{z{^{[1]}}}} \frac{\partial{z{^{[1]}}}}{\bf{b^{[1]}}}=
d{z^{[1]}}^{T}\\\\

& \frac{\partial{E}}{\partial{w}^{[1]}} = \frac{\partial{E}}{\partial{h}} 
\frac{\partial{h}}{\partial{z{^{[1]}}}} \frac{\partial{z{^{[1]}}}}{\bf{W^{[1]}}}=
d{z^{[1]}}^{T}x^{T}
\end{aligned}
$$

---

## 多层神经网络

对于第i层有
$$
\begin{aligned}
\text{正向传播:} \\\\

& \text{输入: } a^{[i-1]} \\
& \text{当前层变量: } W^{[i]},b{[i]},g() \\
& \text{输出: } z^{[i]},a^{[i]} \\\\

& z^{[i]} = W^{[i]} a^{[i-1]} + b^{[i]} \\\\

& a^{[i]} = g(z^{[i]}) \\\\

\text{反向传播:} \\\\

& \text{input:} da^{[i]}\\
& \text{layer params: } z^{[i]},a^{[i]},W^{[i]},b^{[i]},g'() \\
& \text{output: } dW^{[i]},db^{[i]},da^{[i-1]}\\\\

&dz^{[i]} = da^{[i]} \circ g'(z^{[i]}) \\\\
& d{W^{[i]}} = d{z^{[i]}}^{T} {a^{[i-1]}}^{T}\\\\
& d{b^{[i]}} = d{z^{[i]}}^{T} \\\\
& d{a^{[i-1]}} = d{z^{[i]}} w^{[i]}
\end{aligned}
$$

---

## 正则化

- L2 norm

$$
\begin{aligned}

& \text{对于向量 w(logistic regression)  : } ||w||^{2}_{2} = \sum_{i=1}^{n}w_{i}^{2}\\\\
& J(w,b) = \frac{1}{m}\sum_{i=1}^{m}(a^{[i]},y^{[i]}) + \frac{\lambda}{2m}||w||^{2}_{2}
  
\end{aligned}
$$

- Frobenius norm

$$
\begin{aligned}

&\text{对于矩阵 W(nn) $\in{a \times b}$}: ||W||_{\bf{F}}^{2} = \sum_{i=1}^{a}\sum_{j=1}^{b}w_{ij}^{2}\\\\

& J(w^{[1]},b^{[1]},\cdots,w^{[l]},b^{[l]}) = \frac{1}{m} \sum_{i=1}^{m}l(a^{[i]},y^{[i]}) + \frac{\lambda}{2m} \sum_{l=1}^{l}
||W^{[i]}||^{2}_{\bf{F}}\\\\

& \text{正则化后偏导数变为 : } dW^{[i]} = d{z^{[i]}}^{T} {a^{[i-1]}}^{T} + \frac{\lambda}{2m}W^{[i]}\\\\

& W^{[i]} = (1-\frac{\lambda*r}{2m})W^{[i]} - r*d{z^{[i]}}^{T} {a^{[i-1]}}^{T}\\\\
& \text{与正则化之前对比可发现,$W^{[i]}$ 乘了一个略小于1的参数,所有有时也叫权值衰减 weight decay} \\\\

\end{aligned}
$$

- dropout

1. inverted dropout

- normalizing training sets

1. zero out mean 均值归零

$$
\begin{aligned}

&M = \frac{1}{m} \sum_{i=0}^{m}x^{[i]}\\

& X = X - M
\end{aligned}

$$

2. normalize the variances 归一化方差

$$
\begin{aligned}

& \sigma^{2} = \frac{1}{m} \sum_{i=1}^{m}{x^{[i]}}^{2} \\

& X = \frac{X}{\sigma^{2}}
  
\end{aligned}
$$
---

## Gradient checking (Grad check)


---

# 优化算法
## batch gradient descent 
  when the data is large (>=2000) take too match time per iteration
## stochastic gradient descent (随机梯度下降)
  the noise can be ameliorated or reduced by smaller learning rate.
  loss speed up from vectorization
## mini-batch gradient descent
  common mini-batch size : (64, ... ,512)

---

## 指数加权平均数 (exponentially weighted averages)

$$
\begin{aligned}
& v_{t} = \beta v_{t-1} + (1 - \beta) \theta_{t} \\\\
& v_{t} \text{可近似看作} 前\frac{1}{1-\beta} \text{条数据的平均值}
\end{aligned}
$$

## 偏差修正 (bias correction)

$$
\begin{aligned}
& v_{t} = \frac{v_{t}}{1 - \beta^{t}}
\end{aligned}
$$

---

## 动量梯度下降 (gradient descent with momentum)

$$
\begin{aligned}
& V_{dw} = \beta V_{dw} + (1 - \beta)dw \\\\
& V_{db} = \beta V_{db} + (1 - \beta)db \\\\
& W = W - \alpha V_{dw} \\\\
& b = b - \alpha V_{db} \\\\
& \text{commonly : }  \beta = 0.9 \text{  means averaging of the last ten iteractions gradients}
\end{aligned}
$$

## RMSprop (root  mean square prop)

$$
\begin{aligned}
& S_{dw} = \beta S_{dw} + (1 - \beta){dw}^{2} \text{ (element-wise square)} \\\\
& S_{db} = \beta S_{db} + (1 - \beta){db}^{2} \\\\
& W = W - \alpha \frac{dw}{\sqrt{S_{dw}} + \epsilon} \quad (\epsilon = 10^{-8})\\\\
& b = b - \alpha \frac{dw}{\sqrt{S_{db}} + \epsilon} \\\\
\end{aligned}
$$

## Adam optimization algorithm

$$
\begin{aligned}
& V_{dw} = \beta_{1} V_{dw} + (1 - \beta_{1})dw \\\\
& V_{db} = \beta_{1} V_{db} + (1 - \beta_{1})db \\\\

& S_{dw} = \beta_{2} S_{dw} + (1 - \beta_{2}){dw}^{2} \text{ (element-wise square)} \\\\
& S_{db} = \beta_{2} S_{db} + (1 - \beta_{2}){db}^{2} \\\\

& V_{dw}^{c} = \frac{V_{dw}}{1 - \beta_{1}^{2}}\\\\
& V_{db}^{c} = \frac{V_{db}}{1 - \beta_{1}^{2}}\\\\
& S_{dw}^{c} = \frac{S_{dw}}{1 - \beta_{2}^{2}}\\\\
& S_{db}^{c} = \frac{S_{db}}{1 - \beta_{2}^{2}}\\\\

& W = W - \alpha \frac{V_{dw}^{c}}{\sqrt{S_{dw}^{c}} + \epsilon} \quad (\epsilon = 10^{-8})\\\\
& b = b - \alpha \frac{V_{db}^{c}}{\sqrt{S_{db}^{c}} + \epsilon} \\\\

& \text{hyperparameters choice : } \\\\
& \beta_{1} = 0.9 \quad \beta_{2} = 0.999 \quad \epsilon = 10^{-8}


\end{aligned}
$$

---

## 学习率衰减 (learning rate decay)

$$
\begin{aligned}
& \alpha = \frac{1}{1 + \text{decay-rate} \times \text{epoch-num}} \alpha_{0} \\\\

& \text{指数衰减 (exponential decay) : } \alpha = \text{decay-rate}^{\text{epoch-num}} \alpha_{0}\\\\
& \cdots
  
\end{aligned}
$$