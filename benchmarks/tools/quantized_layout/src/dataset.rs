use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

pub(crate) fn read_fvecs(path: &Path, rows: usize, dimension: usize) -> io::Result<Vec<f32>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut vectors = Vec::with_capacity(rows * dimension);
    let mut dimension_bytes = [0_u8; 4];

    for row in 0..rows {
        reader.read_exact(&mut dimension_bytes).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("failed to read fvecs row {row}: {error}"),
            )
        })?;
        let source_dimension = i32::from_le_bytes(dimension_bytes);
        if source_dimension <= 0 || dimension > source_dimension as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "requested dimension {dimension} exceeds fvecs dimension {source_dimension}"
                ),
            ));
        }

        for _ in 0..dimension {
            let mut bytes = [0_u8; 4];
            reader.read_exact(&mut bytes)?;
            vectors.push(f32::from_le_bytes(bytes));
        }
        let skipped = (source_dimension as usize - dimension) * size_of::<f32>();
        if skipped > 0 {
            io::copy(&mut reader.by_ref().take(skipped as u64), &mut io::sink())?;
        }
    }
    Ok(vectors)
}
