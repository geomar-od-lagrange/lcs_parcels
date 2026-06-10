"""Utilities for building regular 2-D particle grids."""


def make_particle_grid(
    *,
    lon_range,
    lat_range,
    dlon,
    dlat,
):
    """Create a regular 2-D particle grid.

    Parameters
    ----------
    lon_range:
        ``(lon_min, lon_max)`` in degrees East.
    lat_range:
        ``(lat_min, lat_max)`` in degrees North.
    dlon:
        Longitude spacing in degrees.
    dlat:
        Latitude spacing in degrees.

    Returns
    -------
    lon, lat :
        Flat 1-D arrays of particle longitudes and latitudes.
        Both arrays have length ``nx * ny`` and are compatible with
        ``ParticleSet.from_list(fieldset, pclass, lon=lon, lat=lat)``.
    """
    pass


def neighbour_indices(
    i,
    j,
    nx,
    ny,
):
    """Return the grid indices of the four cardinal neighbours of particle (i, j).

    Parameters
    ----------
    i:
        Longitude index of the reference particle (0-based, x-axis).
    j:
        Latitude index of the reference particle (0-based, y-axis).
    nx:
        Number of grid points in the longitude (x) direction.
    ny:
        Number of grid points in the latitude (y) direction.

    Returns
    -------
    list of (int, int)
        ``(i, j)`` tuples of existing neighbours in the order
        ``[+x, +y, -x, -y]``. Neighbours outside the grid bounds are
        omitted, so the list may contain fewer than four entries.
    """
    pass
