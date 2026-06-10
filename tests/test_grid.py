from lcs_parcels.grid import make_particle_grid


def test_grid_created():
    lon, lat = make_particle_grid(
        lon_range=[10, 20], lat_range=[10, 20], dlon=1, dlat=1
    )
    assert len(lon) != 0
    assert len(lat) != 0
