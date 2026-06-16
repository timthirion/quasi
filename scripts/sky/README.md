# Hosek-Wilkie data vendor

`fetch_hosek_data.py` downloads the official Hosek-Wilkie 2012 sky
model reference release from cgg.mff.cuni.cz, extracts the RGB
coefficient tables from `ArHosekSkyModelData_RGB.h`, and rewrites the
`mod data` block inside `src/pathtrace/sky.rs` with the real values.

## When to run this

* Once on initial setup, after the PT-sky/hosek-cpu math + tests have
  landed.
* When the upstream zip changes — the script's `EXPECTED_ZIP_SHA256`
  guard makes that surface as a deliberate roll-forward.
* Never as part of CI — the script does network I/O and modifies
  tracked source. CI runs against whatever was committed.

## Workflow

```sh
python3 scripts/sky/fetch_hosek_data.py
cargo fmt --all
cargo test --lib pathtrace::sky
```

The fmt pass cleans up the float literals (rustfmt's formatting of
long array literals is less ugly than what the script emits raw).
The tests should still pass — the math is the math, but you now have
real numbers.

## On the SHA-256 guard

The `EXPECTED_ZIP_SHA256` constant near the top of the script is empty
on first run. After the first successful vendor, replace it with the
SHA reported by the script. From then on, any upstream change has to
be acknowledged before the script accepts new data.

## License

The vendored data is BSD-licensed by Hošek & Wilkie. The generated
Rust file includes the BSD attribution as a comment header.
