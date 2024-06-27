import solafune_tools.make_mosaic
import pytest


def test_mosaic_mode():
    with pytest.raises(ValueError) as expected_error:
        solafune_tools.make_mosaic.create_mosaic(mosaic_mode="Mode")
    assert (
        str(expected_error.value)
        == "Please use either 'Median' or 'Minimum' only for the parameter 'mosaic_mode'"
    )
