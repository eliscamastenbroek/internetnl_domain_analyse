from internetnl_domain_analyse import

__author__ = "Eelco van Vliet"
__copyright__ = "Eelco van Vliet"
__license__ = "MIT"


def test_main(capsys):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserting against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    weighted_sample_statistics = WeightedSampleStatistics(
        group_keys=None, records_df_selection=None, weights_df=None
    )
    assert weighted_sample_statistics.group_keys is None
