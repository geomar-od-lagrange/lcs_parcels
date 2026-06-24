- Follow Haller (2015) for naming and notation. (Use /zotero skill)

- Data Structures:
  - particle grid
    - particle grid
      - xr dataset w/ i j dims and lon(i,j) and lat(i,j) data vars
      - these are the locations on which FTLE etc. will be defined
      - resolution dlon, dlat
    - optional: displacement grid
      - dims (di, dj)
      - displacement dx dy
  - particle set
    - stacked i,j or i,j,di,dj dims
  - roundtrip necessary (ie need constructor of particle grid from particle set)

- operators
  - dX/dy etc
  - F (2x2 on ij grid)
  - C (2x2 on ij grid)
  - eigen (C) on 2xij grid
  - more derived from eigen analysis

- higher order
  - LCS construction based on xi_1,2