- Ellipsoid is implemented as an intersectable form its implicit equation: ax^2 + by^2 + cz^2 -1 = 0

- The normal to ellipsoid is its gradient equation: ax + by + cz = 0

- The dodecahedron is implemented as a series of intersected planes

- The dodecahedron diffuse part is a function of threshold between closest edge and hit position

- Indexes in OBJ file start from 1 while in C++ it starts from zero, that was taken in account
