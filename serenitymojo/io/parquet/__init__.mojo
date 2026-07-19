# serenitymojo/io/parquet — pure-Mojo Apache Parquet reader for dataset ingestion.
#
# Reads the flat Parquet shards ML datasets ship as: BYTE_ARRAY columns holding
# caption strings + inline media blobs (PNG/JPEG/MP4 bytes), SNAPPY codec, PLAIN
# and RLE_DICTIONARY encodings. Feeds the trainer's cache builders via extract.
# No Python, no Arrow, no C. See MAP.md / docs/MOJO_MODULES.md.
