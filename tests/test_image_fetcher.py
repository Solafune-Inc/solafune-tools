import os

import solafune_tools


def test_get_stac_defaults():
    gdf = solafune_tools.make_stac_query(prefix="test")
    if os.path.isfile(gdf):
        os.remove(gdf)
        assert True
    else:
        assert False
