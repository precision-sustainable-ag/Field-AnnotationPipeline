import logging
from pathlib import Path

from src.utils.cvat_scripts.package_cvat import PackageCVAT


def test_no_images(tmp_path, caplog):
    species_dir = tmp_path / "species"
    cutouts = species_dir / "cutouts"
    cutouts.mkdir(parents=True)

    package = PackageCVAT(species_dir)

    caplog.set_level(logging.WARNING)
    package.process_images_and_masks()

    expected_zip = species_dir / f"{package.species}_cvat_package.zip"
    assert not expected_zip.exists()
    assert any(
        rec.levelno == logging.WARNING and "No images found" in rec.message
        for rec in caplog.records
    )
