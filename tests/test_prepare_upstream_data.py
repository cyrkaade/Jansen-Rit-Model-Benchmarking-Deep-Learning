import hashlib

from scripts import prepare_upstream_data


def test_prepare_dataset_downloads_without_git_lfs(tmp_path, monkeypatch):
    payload = b"small deterministic dataset"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(
        prepare_upstream_data.DATASETS,
        "test",
        {
            "source": "unused",
            "destination": "data/test.bin",
            "sha256": expected_hash,
            "bytes": len(payload),
        },
    )
    monkeypatch.setattr(prepare_upstream_data.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        prepare_upstream_data,
        "SOURCE_REPOSITORY",
        "unused",
    )
    monkeypatch.setattr(
        prepare_upstream_data,
        "SOURCE_COMMIT",
        "unused",
    )
    monkeypatch.setattr(
        prepare_upstream_data,
        "_download",
        lambda _url, destination, _size: destination.write_bytes(payload),
    )

    destination = prepare_upstream_data.prepare_dataset(tmp_path, "test")

    assert destination.read_bytes() == payload
    assert destination == tmp_path / "data" / "test.bin"


def test_download_streams_to_verified_size(tmp_path):
    payload = b"streamed data"
    source = tmp_path / "remote.bin"
    destination = tmp_path / "downloaded.bin"
    source.write_bytes(payload)

    prepare_upstream_data._download(source.as_uri(), destination, len(payload))

    assert destination.read_bytes() == payload
    assert not destination.with_suffix(".bin.part").exists()
