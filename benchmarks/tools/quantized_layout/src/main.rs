mod benchmark;
mod cli;
mod dataset;
mod layout;
mod quantization;
mod simd;

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;

use benchmark::{BenchmarkConfig, BenchmarkReport, benchmark_case};
use cli::Args;
use dataset::read_fvecs;
use quantization::{Quantizer, QuantizerKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.output_dir)?;

    eprintln!(
        "loading {} rows x {} dimensions from {} (outside timing)",
        args.rows,
        args.dimension,
        args.input.display()
    );
    let vectors = read_fvecs(&args.input, args.rows, args.dimension)?;
    let query = vectors[..args.dimension].to_vec();

    let config = BenchmarkConfig {
        rows: args.rows,
        dimension: args.dimension,
        batch_rows: args.batch_rows,
        candidate_rows: args.candidate_rows.clone(),
        warmups: args.warmups,
        repetitions: args.repetitions,
    };

    let mut cases = Vec::new();
    for kind in QuantizerKind::ALL {
        eprintln!("preparing {}", kind.name());
        let quantizer = Quantizer::train(kind, &vectors, args.rows, args.dimension);
        let encoded = quantizer.encode(&vectors, args.rows, args.dimension);
        cases.push(benchmark_case(
            &args.output_dir,
            &config,
            &quantizer,
            &encoded,
            &query,
        )?);
    }

    let report = BenchmarkReport {
        input: args.input,
        distance_kernel: simd::backend_name(),
        config,
        cases,
    };
    let output = args
        .output
        .unwrap_or_else(|| PathBuf::from(&args.output_dir).join("result.json"));
    let writer = BufWriter::new(File::create(&output)?);
    serde_json::to_writer_pretty(writer, &report)?;
    eprintln!("wrote {}", output.display());
    Ok(())
}
