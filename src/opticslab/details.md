## What these file teaches:

•	 ray.py : a ray is two numbers, its height  y  above the optical axis and its slope  u . The “paraxial” simplification means rays stay near the axis and slightly tilted, so the maths reduces to multiplication and addition.

•	 elements.py : each element is a 2×2 ABCD matrix. Free space is  [[1, d], [0, 1]]  and a thin lens is  [[1, 0], [-1/f, 1]] . An aperture blocks any ray with |y| larger than its radius.

•	 system.py : a system is a list of elements. Multiplying their matrices gives one system matrix, with the last element on the left.  trace()  records  z ,  y  and  u  after each element.

•	 analysis.py : the effective focal length is  -1/C . The image forms where the matrix element  B  is zero, because then every ray from one object point lands at the same height.
