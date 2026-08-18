import pytest

@pytest.mark.metal
def test_incremental_mtp_matches_prefill():
    """Integration parity test; run on Apple Silicon with --model fixture.

    Kept as a marker until CI supplies a Qwythos checkpoint and Metal device.
    """
    pytest.skip("requires Qwythos fixture and Metal; invoke the parity harness")
