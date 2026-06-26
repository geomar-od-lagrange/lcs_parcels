- Follow Haller (2015) for naming and notation. (Use /zotero skill)

- Data Structures:
  - particle grid
    - particle grid
      - xr dataset w/ i j dims
      - reference coords lon_0(i,j), lat_0(i,j) = x_0 (where FTLE etc. is defined)
      - advected data vars lon(i,j), lat(i,j) = F(x_0)
      - resolution dlon, dlat
    - optional: auxiliary displacement stencil
      - single displacement dim = [east, north, west, south]
      - offsets dx, dy in meters (fixed four arms; no center, no diagonals)
  - particle set
    - stacked i,j or i,j,displacement dims
  - roundtrip necessary (ie need constructor of particle grid from particle set)

- operators
  - dX/dy etc
  - F (2x2 on ij grid)
  - C (2x2 on ij grid)
  - eigen (C) on 2xij grid
  - more derived from eigen analysis

- higher order
  - LCS construction based on xi_1,2