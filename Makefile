VERSION := $(shell python -c "from src import config; print(config.__version__)")
OUTPUT_DIR=data/output/$(VERSION)

$(OUTPUT_DIR)/country_stats.geojson: src/country_stats.py $(OUTPUT_DIR)/contiguous_hotspots.gpkg
	python $<

$(OUTPUT_DIR)/contiguous_hotspots.gpkg: src/contiguous_hotspots.py
	python $^
