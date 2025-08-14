from pathlib import Path
from s3fs import S3FileSystem

VERSION = "0-0-1"


def upload_file(file, path):
    s3 = S3FileSystem()
    s3.put(file, path)


def main():
    input_folder = Path("data/output")
    output_folder = Path(
        f"dep-public-staging/dep_ls_coastlines/dashboard_stats/{VERSION}"
    )

    upload_file(input_folder / "contiguous_hotspots.gpkg", output_folder)
    upload_file(input_folder / "contiguous_hotspots.pmtiles", output_folder)
    upload_file(input_folder / "country_summaries.geojson", output_folder)


if __name__ == "__main__":
    main()
