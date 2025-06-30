from omniweb_downloader import OMNIWebDownloader

def test_creation():
    d = OMNIWebDownloader(variables=["sym_h"])
    assert isinstance(d, OMNIWebDownloader)
