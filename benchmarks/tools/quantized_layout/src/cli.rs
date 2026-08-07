use std::env;
use std::path::PathBuf;

pub(crate) struct Args {
    pub(crate) input: PathBuf,
    pub(crate) output_dir: PathBuf,
    pub(crate) output: Option<PathBuf>,
    pub(crate) rows: usize,
    pub(crate) dimension: usize,
    pub(crate) batch_rows: usize,
    pub(crate) candidate_rows: Vec<usize>,
    pub(crate) warmups: usize,
    pub(crate) repetitions: usize,
}

impl Args {
    pub(crate) fn parse() -> Result<Self, String> {
        let mut input = None;
        let mut output_dir = PathBuf::from("target/quantized-layout");
        let mut output = None;
        let mut rows = 262_144;
        let mut dimension = 384;
        let mut batch_rows = 8_192;
        let mut candidate_rows = vec![8_192, 65_536, 262_144];
        let mut warmups = 3;
        let mut repetitions = 10;

        let mut arguments = env::args().skip(1);
        while let Some(argument) = arguments.next() {
            let value = |arguments: &mut std::iter::Skip<std::env::Args>| {
                arguments
                    .next()
                    .ok_or_else(|| format!("missing value for {argument}"))
            };
            match argument.as_str() {
                "--input" => input = Some(PathBuf::from(value(&mut arguments)?)),
                "--output-dir" => output_dir = PathBuf::from(value(&mut arguments)?),
                "--output" => output = Some(PathBuf::from(value(&mut arguments)?)),
                "--rows" => rows = parse_positive(&value(&mut arguments)?, "rows")?,
                "--dimension" => {
                    dimension = parse_positive(&value(&mut arguments)?, "dimension")?;
                }
                "--batch-rows" => {
                    batch_rows = parse_positive(&value(&mut arguments)?, "batch rows")?;
                }
                "--candidate-rows" => {
                    candidate_rows = value(&mut arguments)?
                        .split(',')
                        .map(|part| parse_positive(part, "candidate rows"))
                        .collect::<Result<Vec<_>, _>>()?;
                }
                "--warmups" => warmups = parse_positive(&value(&mut arguments)?, "warmups")?,
                "--repetitions" => {
                    repetitions = parse_positive(&value(&mut arguments)?, "repetitions")?;
                }
                "--help" | "-h" => return Err(usage()),
                _ => return Err(format!("unknown argument: {argument}\n\n{}", usage())),
            }
        }

        let input = input.ok_or_else(usage)?;
        candidate_rows.sort_unstable();
        candidate_rows.dedup();
        if candidate_rows.last().is_some_and(|&value| value > rows) {
            return Err("candidate rows must not exceed --rows".into());
        }

        Ok(Self {
            input,
            output_dir,
            output,
            rows,
            dimension,
            batch_rows,
            candidate_rows,
            warmups,
            repetitions,
        })
    }
}

fn parse_positive(value: &str, name: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name}: {error}"))?;
    if parsed == 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(parsed)
}

fn usage() -> String {
    "usage: cargo run --release -- --input BASE.fvecs [--rows 262144] \\\n+  [--dimension 384] [--batch-rows 8192] \\\n+  [--candidate-rows 8192,65536,262144] [--warmups 3] [--repetitions 10] \\\n+  [--output-dir PATH] [--output PATH]"
        .into()
}
