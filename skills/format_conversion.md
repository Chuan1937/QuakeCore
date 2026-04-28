# Format Conversion

Use this skill when you want to convert seismic data between supported formats.

## Typical workflow

1. Identify the source file format.
2. Confirm the destination format and naming convention.
3. Check whether metadata should be preserved or transformed.
4. Validate the converted file before using it downstream.

## Common conversions

- MiniSEED to SEGY
- SEGY to MiniSEED
- SAC to MiniSEED
- HDF5 to analysis-ready output

## Validation checklist

- File opens successfully after conversion
- Trace count matches expectations
- Timing metadata remains sensible
- The destination file is written to the intended directory

## Notes

- Keep an eye on coordinate metadata and channel names.
- Confirm sampling intervals after conversion.
