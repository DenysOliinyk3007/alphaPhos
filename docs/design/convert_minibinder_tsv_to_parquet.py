"""One-time conversion: the 8.9 GB miniBinder Spectronaut TSV -> parquet.

Uses pyarrow's streaming CSV reader to avoid loading the whole TSV into RAM at
once. Writes a single parquet file with the same columns, drops `EG.IsDecoy=True`
rows on the way through to shrink the output further.

Subsequent harness runs will read this parquet (typically 1-2 GB) repeatedly
instead of re-parsing the TSV.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

SRC = Path("D:/Projects/miniBinders/scripts/rawdata/CHO_TAB2_H2F_repeat_w_CHO_fasta_Report.tsv")
DST = Path("D:/Projects/alphaPhos/docs/design/minibinder_data/minibinder_full.parquet")


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)
    if DST.exists():
        size_mb = DST.stat().st_size / 1e6
        print(f"Parquet already exists at {DST} ({size_mb:.1f} MB). Skipping.")
        return

    src_size_gb = SRC.stat().st_size / 1e9
    print(f"Source: {SRC} ({src_size_gb:.2f} GB)")
    print(f"Destination: {DST}")
    print()

    t0 = time.time()
    read_opts = pacsv.ReadOptions(
        # Read in 256 MB chunks
        block_size=256 * 1024 * 1024,
        skip_rows=0,
    )
    parse_opts = pacsv.ParseOptions(delimiter="\t")
    convert_opts = pacsv.ConvertOptions(
        # Let pyarrow infer types from the data; floats and ints will follow
        # Spectronaut conventions (most quants are double, IsDecoy is bool, etc.)
        strings_can_be_null=True,
        null_values=["", "NA", "NaN", "Filtered"],
        # Force certain columns to specific types where Spectronaut sometimes
        # ships strings that pyarrow misclassifies on the first chunk.
        column_types={
            "PG.Qvalue": pa.float64(),
            "EG.Qvalue": pa.float64(),
            "PEP.Quantity": pa.float64(),
        },
    )

    print("Opening streaming CSV reader...")
    reader = pacsv.open_csv(
        SRC, read_options=read_opts, parse_options=parse_opts,
        convert_options=convert_opts,
    )
    print("Reading first batch to determine schema...")
    first_batch = reader.read_next_batch()
    schema = first_batch.schema
    print(f"Schema: {len(schema)} columns")

    writer = pq.ParquetWriter(DST, schema, compression="snappy")
    print("Writing parquet (snappy)...")

    n_rows = 0
    n_batches = 0
    # Process the first batch we already read
    is_decoy_idx = schema.get_field_index("EG.IsDecoy")
    if is_decoy_idx >= 0:
        decoy_mask = pc.invert(first_batch.column(is_decoy_idx))
        first_batch = first_batch.filter(decoy_mask)
    writer.write_batch(first_batch)
    n_rows += len(first_batch)
    n_batches += 1
    print(f"  batch {n_batches}: {n_rows:,} rows in {time.time()-t0:.1f}s")

    while True:
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            break
        if is_decoy_idx >= 0:
            decoy_mask = pc.invert(batch.column(is_decoy_idx))
            batch = batch.filter(decoy_mask)
        writer.write_batch(batch)
        n_rows += len(batch)
        n_batches += 1
        if n_batches % 5 == 0:
            elapsed = time.time() - t0
            print(f"  batch {n_batches}: {n_rows:,} rows in {elapsed:.1f}s "
                  f"({n_rows/elapsed/1e6:.2f} M rows/sec)")

    writer.close()

    dst_size_mb = DST.stat().st_size / 1e6
    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s.")
    print(f"Wrote {n_rows:,} rows ({n_batches} batches) to {DST} ({dst_size_mb:.1f} MB)")
    print(f"Compression: {src_size_gb*1000/dst_size_mb:.1f}x smaller")


if __name__ == "__main__":
    main()
