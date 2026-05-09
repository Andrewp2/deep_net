use clap::{Parser, Subcommand};
use flate2::read::GzDecoder;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

const INPUTS: usize = 28 * 28;
const WIDTH: usize = 2;
const CLASSES: usize = 10;
const LAYER_FLOATS: usize = 6;
const LAYER_BYTES: u64 = (LAYER_FLOATS * std::mem::size_of::<f32>()) as u64;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(version, about = "Disk-backed absurdly deep width-2 MNIST experiment")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create a binary layer file. 100M width-2 layers is 2.4 GB.
    Init {
        #[arg(long, default_value_t = 100_000_000)]
        layers: u64,
        #[arg(long, default_value = "weights.bin")]
        weights: PathBuf,
        #[arg(long, default_value_t = 1_000_000)]
        chunk_layers: usize,
        #[arg(long, default_value_t = 0x5EED)]
        seed: u64,
        #[arg(long, default_value_t = 0.001)]
        scale: f32,
    },
    /// Run exact chunk-checkpointed SGD on MNIST.
    Train {
        #[arg(long, default_value_t = 100_000_000)]
        layers: u64,
        #[arg(long, default_value = "weights.bin")]
        weights: PathBuf,
        #[arg(long, default_value = "head.bin")]
        head: PathBuf,
        #[arg(long, default_value = "data/mnist")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        steps: usize,
        #[arg(long, default_value_t = 1)]
        batch: usize,
        #[arg(long, default_value_t = 1_000_000)]
        chunk_layers: usize,
        #[arg(long)]
        alpha: Option<f32>,
        #[arg(long, default_value_t = 0.05)]
        layer_lr: f32,
        #[arg(long, default_value_t = 0.01)]
        head_lr: f32,
        #[arg(long, default_value_t = 0xC0FFEE)]
        seed: u64,
        #[arg(long, default_value_t = 60_000)]
        train_limit: usize,
        #[arg(long, default_value_t = true)]
        init_if_missing: bool,
    },
    /// Forward-only accuracy check. Use small sample counts for huge depths.
    Eval {
        #[arg(long, default_value_t = 100_000_000)]
        layers: u64,
        #[arg(long, default_value = "weights.bin")]
        weights: PathBuf,
        #[arg(long, default_value = "head.bin")]
        head: PathBuf,
        #[arg(long, default_value = "data/mnist")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 100)]
        samples: usize,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 1_000_000)]
        chunk_layers: usize,
        #[arg(long)]
        alpha: Option<f32>,
    },
    /// Create a dense variable-width layer file.
    InitDense {
        #[arg(long, default_value_t = 1_000_000)]
        layers: u64,
        #[arg(long, default_value_t = 32)]
        width: usize,
        #[arg(long, default_value = "dense_weights.bin")]
        weights: PathBuf,
        #[arg(long, default_value_t = 100_000)]
        chunk_layers: usize,
        #[arg(long, default_value_t = 0xD00D)]
        seed: u64,
        #[arg(long, default_value_t = 0.001)]
        scale: f32,
    },
    /// Train a dense variable-width residual core on MNIST.
    TrainDense {
        #[arg(long, default_value_t = 1_000_000)]
        layers: u64,
        #[arg(long, default_value_t = 32)]
        width: usize,
        #[arg(long, default_value = "dense_weights.bin")]
        weights: PathBuf,
        #[arg(long, default_value = "dense_head.bin")]
        head: PathBuf,
        #[arg(long, default_value = "data/mnist")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        steps: usize,
        #[arg(long, default_value_t = 1)]
        batch: usize,
        #[arg(long, default_value_t = 100_000)]
        chunk_layers: usize,
        #[arg(long)]
        alpha: Option<f32>,
        #[arg(long, default_value_t = 0.01)]
        layer_lr: f32,
        #[arg(long, default_value_t = 0.01)]
        head_lr: f32,
        #[arg(long, default_value_t = 0xC0FFEE)]
        seed: u64,
        #[arg(long, default_value_t = 60_000)]
        train_limit: usize,
        #[arg(long, default_value_t = true)]
        init_if_missing: bool,
        #[arg(long, default_value_t = 1)]
        report_every: usize,
    },
    /// Evaluate a dense variable-width residual core on MNIST.
    EvalDense {
        #[arg(long, default_value_t = 1_000_000)]
        layers: u64,
        #[arg(long, default_value_t = 32)]
        width: usize,
        #[arg(long, default_value = "dense_weights.bin")]
        weights: PathBuf,
        #[arg(long, default_value = "dense_head.bin")]
        head: PathBuf,
        #[arg(long, default_value = "data/mnist")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 100)]
        samples: usize,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 100_000)]
        chunk_layers: usize,
        #[arg(long)]
        alpha: Option<f32>,
    },
    /// Train a practical MNIST MLP head for low loss.
    TrainMlp {
        #[arg(long, default_value = "mlp_head.bin")]
        head: PathBuf,
        #[arg(long, default_value = "data/mnist")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 128)]
        hidden: usize,
        #[arg(long, default_value_t = 1_000)]
        steps: usize,
        #[arg(long, default_value_t = 128)]
        batch: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, default_value_t = 0xABCDEF)]
        seed: u64,
        #[arg(long, default_value_t = 60_000)]
        train_limit: usize,
        #[arg(long, default_value_t = 1000)]
        report_every: usize,
    },
    /// Evaluate the practical MNIST MLP head.
    EvalMlp {
        #[arg(long, default_value = "mlp_head.bin")]
        head: PathBuf,
        #[arg(long, default_value = "data/mnist")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 128)]
        hidden: usize,
        #[arg(long, default_value_t = 10_000)]
        samples: usize,
        #[arg(long, default_value_t = 256)]
        batch: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Init {
            layers,
            weights,
            chunk_layers,
            seed,
            scale,
        } => init_weights(&weights, layers, chunk_layers, seed, scale),
        Command::Train {
            layers,
            weights,
            head,
            data_dir,
            steps,
            batch,
            chunk_layers,
            alpha,
            layer_lr,
            head_lr,
            seed,
            train_limit,
            init_if_missing,
        } => train(TrainConfig {
            layers,
            weights,
            head,
            data_dir,
            steps,
            batch,
            chunk_layers,
            alpha: alpha.unwrap_or_else(|| default_alpha(layers)),
            layer_lr,
            head_lr,
            seed,
            train_limit,
            init_if_missing,
        }),
        Command::Eval {
            layers,
            weights,
            head,
            data_dir,
            samples,
            batch,
            chunk_layers,
            alpha,
        } => eval(EvalConfig {
            layers,
            weights,
            head,
            data_dir,
            samples,
            batch,
            chunk_layers,
            alpha: alpha.unwrap_or_else(|| default_alpha(layers)),
        }),
        Command::InitDense {
            layers,
            width,
            weights,
            chunk_layers,
            seed,
            scale,
        } => init_dense_weights(&weights, layers, width, chunk_layers, seed, scale),
        Command::TrainDense {
            layers,
            width,
            weights,
            head,
            data_dir,
            steps,
            batch,
            chunk_layers,
            alpha,
            layer_lr,
            head_lr,
            seed,
            train_limit,
            init_if_missing,
            report_every,
        } => train_dense(DenseTrainConfig {
            layers,
            width,
            weights,
            head,
            data_dir,
            steps,
            batch,
            chunk_layers,
            alpha: alpha.unwrap_or_else(|| default_alpha(layers)),
            layer_lr,
            head_lr,
            seed,
            train_limit,
            init_if_missing,
            report_every,
        }),
        Command::EvalDense {
            layers,
            width,
            weights,
            head,
            data_dir,
            samples,
            batch,
            chunk_layers,
            alpha,
        } => eval_dense(DenseEvalConfig {
            layers,
            width,
            weights,
            head,
            data_dir,
            samples,
            batch,
            chunk_layers,
            alpha: alpha.unwrap_or_else(|| default_alpha(layers)),
        }),
        Command::TrainMlp {
            head,
            data_dir,
            hidden,
            steps,
            batch,
            lr,
            seed,
            train_limit,
            report_every,
        } => train_mlp(MlpTrainConfig {
            head,
            data_dir,
            hidden,
            steps,
            batch,
            lr,
            seed,
            train_limit,
            report_every,
        }),
        Command::EvalMlp {
            head,
            data_dir,
            hidden,
            samples,
            batch,
        } => eval_mlp(MlpEvalConfig {
            head,
            data_dir,
            hidden,
            samples,
            batch,
        }),
    }
}

fn default_alpha(layers: u64) -> f32 {
    1.0 / (layers as f32).sqrt()
}

struct TrainConfig {
    layers: u64,
    weights: PathBuf,
    head: PathBuf,
    data_dir: PathBuf,
    steps: usize,
    batch: usize,
    chunk_layers: usize,
    alpha: f32,
    layer_lr: f32,
    head_lr: f32,
    seed: u64,
    train_limit: usize,
    init_if_missing: bool,
}

struct EvalConfig {
    layers: u64,
    weights: PathBuf,
    head: PathBuf,
    data_dir: PathBuf,
    samples: usize,
    batch: usize,
    chunk_layers: usize,
    alpha: f32,
}

struct DenseTrainConfig {
    layers: u64,
    width: usize,
    weights: PathBuf,
    head: PathBuf,
    data_dir: PathBuf,
    steps: usize,
    batch: usize,
    chunk_layers: usize,
    alpha: f32,
    layer_lr: f32,
    head_lr: f32,
    seed: u64,
    train_limit: usize,
    init_if_missing: bool,
    report_every: usize,
}

struct DenseEvalConfig {
    layers: u64,
    width: usize,
    weights: PathBuf,
    head: PathBuf,
    data_dir: PathBuf,
    samples: usize,
    batch: usize,
    chunk_layers: usize,
    alpha: f32,
}

struct MlpTrainConfig {
    head: PathBuf,
    data_dir: PathBuf,
    hidden: usize,
    steps: usize,
    batch: usize,
    lr: f32,
    seed: u64,
    train_limit: usize,
    report_every: usize,
}

struct MlpEvalConfig {
    head: PathBuf,
    data_dir: PathBuf,
    hidden: usize,
    samples: usize,
    batch: usize,
}

#[derive(Clone)]
struct Head {
    input_w: Vec<f32>,
    input_b: [f32; WIDTH],
    output_w: [f32; WIDTH * CLASSES],
    output_b: [f32; CLASSES],
}

impl Head {
    const FLOATS: usize = INPUTS * WIDTH + WIDTH + WIDTH * CLASSES + CLASSES;

    fn load_or_init(path: &Path, seed: u64, scale: f32) -> Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            let head = Self::random(seed, scale);
            head.save(path)?;
            Ok(head)
        }
    }

    fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() != Self::FLOATS * std::mem::size_of::<f32>() {
            return Err(format!(
                "{} has {} bytes, expected {}",
                path.display(),
                bytes.len(),
                Self::FLOATS * std::mem::size_of::<f32>()
            )
            .into());
        }

        let mut floats = Vec::with_capacity(Self::FLOATS);
        for chunk in bytes.chunks_exact(4) {
            floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }

        let mut offset = 0;
        let input_w = floats[offset..offset + INPUTS * WIDTH].to_vec();
        offset += INPUTS * WIDTH;

        let input_b = [floats[offset], floats[offset + 1]];
        offset += WIDTH;

        let mut output_w = [0.0; WIDTH * CLASSES];
        output_w.copy_from_slice(&floats[offset..offset + WIDTH * CLASSES]);
        offset += WIDTH * CLASSES;

        let mut output_b = [0.0; CLASSES];
        output_b.copy_from_slice(&floats[offset..offset + CLASSES]);

        Ok(Self {
            input_w,
            input_b,
            output_w,
            output_b,
        })
    }

    fn random(seed: u64, scale: f32) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut input_w = vec![0.0; INPUTS * WIDTH];
        for w in &mut input_w {
            *w = rng.gen_range(-scale..scale);
        }

        let mut output_w = [0.0; WIDTH * CLASSES];
        for w in &mut output_w {
            *w = rng.gen_range(-scale..scale);
        }

        Self {
            input_w,
            input_b: [0.0; WIDTH],
            output_w,
            output_b: [0.0; CLASSES],
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let mut bytes = Vec::with_capacity(Self::FLOATS * std::mem::size_of::<f32>());
        for value in &self.input_w {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in self.input_b {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in self.output_w {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in self.output_b {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        fs::write(path, bytes)?;
        Ok(())
    }
}

#[derive(Clone)]
struct DenseHead {
    width: usize,
    input_w: Vec<f32>,
    input_b: Vec<f32>,
    output_w: Vec<f32>,
    output_b: [f32; CLASSES],
}

impl DenseHead {
    fn floats(width: usize) -> usize {
        INPUTS * width + width + width * CLASSES + CLASSES
    }

    fn load_or_init(path: &Path, width: usize, seed: u64, scale: f32) -> Result<Self> {
        if path.exists() {
            Self::load(path, width)
        } else {
            let head = Self::random(width, seed, scale);
            head.save(path)?;
            Ok(head)
        }
    }

    fn load(path: &Path, width: usize) -> Result<Self> {
        validate_width(width)?;

        let expected_bytes = Self::floats(width) * std::mem::size_of::<f32>();
        let bytes = fs::read(path)?;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "{} has {} bytes, expected {} for width={}",
                path.display(),
                bytes.len(),
                expected_bytes,
                width
            )
            .into());
        }

        let mut floats = Vec::with_capacity(Self::floats(width));
        for chunk in bytes.chunks_exact(4) {
            floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }

        let mut offset = 0;
        let input_w = floats[offset..offset + INPUTS * width].to_vec();
        offset += INPUTS * width;
        let input_b = floats[offset..offset + width].to_vec();
        offset += width;
        let output_w = floats[offset..offset + width * CLASSES].to_vec();
        offset += width * CLASSES;
        let mut output_b = [0.0; CLASSES];
        output_b.copy_from_slice(&floats[offset..offset + CLASSES]);

        Ok(Self {
            width,
            input_w,
            input_b,
            output_w,
            output_b,
        })
    }

    fn random(width: usize, seed: u64, scale: f32) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let input_scale = (6.0f32 / (INPUTS + width) as f32).sqrt();
        let output_scale = (6.0f32 / (width + CLASSES) as f32).sqrt();

        let mut input_w = vec![0.0; INPUTS * width];
        for w in &mut input_w {
            *w = rng.gen_range(-input_scale..input_scale);
        }

        let mut output_w = vec![0.0; width * CLASSES];
        for w in &mut output_w {
            *w = rng.gen_range(-output_scale..output_scale);
        }

        if scale != 1.0 {
            for w in &mut input_w {
                *w *= scale;
            }
            for w in &mut output_w {
                *w *= scale;
            }
        }

        Self {
            width,
            input_w,
            input_b: vec![0.0; width],
            output_w,
            output_b: [0.0; CLASSES],
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let mut bytes = Vec::with_capacity(Self::floats(self.width) * std::mem::size_of::<f32>());
        for value in &self.input_w {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in &self.input_b {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in &self.output_w {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in self.output_b {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        fs::write(path, bytes)?;
        Ok(())
    }
}

#[derive(Clone)]
struct MlpHead {
    hidden: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: [f32; CLASSES],
}

impl MlpHead {
    fn floats(hidden: usize) -> usize {
        INPUTS * hidden + hidden + hidden * CLASSES + CLASSES
    }

    fn load_or_init(path: &Path, hidden: usize, seed: u64) -> Result<Self> {
        if path.exists() {
            Self::load(path, hidden)
        } else {
            let head = Self::random(hidden, seed);
            head.save(path)?;
            Ok(head)
        }
    }

    fn load(path: &Path, hidden: usize) -> Result<Self> {
        if hidden == 0 {
            return Err("--hidden must be at least 1".into());
        }

        let expected_bytes = Self::floats(hidden) * std::mem::size_of::<f32>();
        let bytes = fs::read(path)?;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "{} has {} bytes, expected {} for hidden={}",
                path.display(),
                bytes.len(),
                expected_bytes,
                hidden
            )
            .into());
        }

        let mut floats = Vec::with_capacity(Self::floats(hidden));
        for chunk in bytes.chunks_exact(4) {
            floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }

        let mut offset = 0;
        let w1 = floats[offset..offset + INPUTS * hidden].to_vec();
        offset += INPUTS * hidden;
        let b1 = floats[offset..offset + hidden].to_vec();
        offset += hidden;
        let w2 = floats[offset..offset + hidden * CLASSES].to_vec();
        offset += hidden * CLASSES;
        let mut b2 = [0.0; CLASSES];
        b2.copy_from_slice(&floats[offset..offset + CLASSES]);

        Ok(Self {
            hidden,
            w1,
            b1,
            w2,
            b2,
        })
    }

    fn random(hidden: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let w1_scale = (6.0f32 / (INPUTS + hidden) as f32).sqrt();
        let w2_scale = (6.0f32 / (hidden + CLASSES) as f32).sqrt();

        let mut w1 = vec![0.0; INPUTS * hidden];
        for w in &mut w1 {
            *w = rng.gen_range(-w1_scale..w1_scale);
        }

        let mut w2 = vec![0.0; hidden * CLASSES];
        for w in &mut w2 {
            *w = rng.gen_range(-w2_scale..w2_scale);
        }

        Self {
            hidden,
            w1,
            b1: vec![0.0; hidden],
            w2,
            b2: [0.0; CLASSES],
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let mut bytes = Vec::with_capacity(Self::floats(self.hidden) * std::mem::size_of::<f32>());
        for value in &self.w1 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in &self.b1 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in &self.w2 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in self.b2 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        fs::write(path, bytes)?;
        Ok(())
    }
}

struct Mnist {
    train_images: Vec<f32>,
    train_labels: Vec<u8>,
    test_images: Vec<f32>,
    test_labels: Vec<u8>,
}

impl Mnist {
    fn load(data_dir: &Path) -> Result<Self> {
        fs::create_dir_all(data_dir)?;

        let train_images = load_idx_file(
            data_dir,
            "train-images-idx3-ubyte",
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        )?;
        let train_labels = load_idx_file(
            data_dir,
            "train-labels-idx1-ubyte",
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        )?;
        let test_images = load_idx_file(
            data_dir,
            "t10k-images-idx3-ubyte",
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        )?;
        let test_labels = load_idx_file(
            data_dir,
            "t10k-labels-idx1-ubyte",
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        )?;

        let (train_images, train_count) = parse_images(&train_images)?;
        let train_labels = parse_labels(&train_labels)?;
        let (test_images, test_count) = parse_images(&test_images)?;
        let test_labels = parse_labels(&test_labels)?;

        if train_count != train_labels.len() {
            return Err("MNIST train image/label count mismatch".into());
        }
        if test_count != test_labels.len() {
            return Err("MNIST test image/label count mismatch".into());
        }

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    fn train_len(&self) -> usize {
        self.train_labels.len()
    }

    fn test_len(&self) -> usize {
        self.test_labels.len()
    }
}

fn train(cfg: TrainConfig) -> Result<()> {
    if cfg.batch == 0 {
        return Err("--batch must be at least 1".into());
    }
    if cfg.chunk_layers == 0 {
        return Err("--chunk-layers must be at least 1".into());
    }

    if !cfg.weights.exists() {
        if cfg.init_if_missing {
            init_weights(&cfg.weights, cfg.layers, cfg.chunk_layers, cfg.seed, 0.001)?;
        } else {
            return Err(format!("{} does not exist", cfg.weights.display()).into());
        }
    }

    verify_weights_len(&cfg.weights, cfg.layers)?;

    let mnist = Mnist::load(&cfg.data_dir)?;
    let mut head = Head::load_or_init(&cfg.head, cfg.seed ^ 0xBAD5EED, 0.05)?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&cfg.weights)?;

    let train_len = cfg.train_limit.min(mnist.train_len());
    if train_len == 0 {
        return Err("no MNIST training samples selected".into());
    }

    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    println!(
        "train layers={} width=2 alpha={} batch={} chunk_layers={} train_len={}",
        cfg.layers, cfg.alpha, cfg.batch, cfg.chunk_layers, train_len
    );

    for step in 0..cfg.steps {
        let mut indices = Vec::with_capacity(cfg.batch);
        for _ in 0..cfg.batch {
            indices.push(rng.gen_range(0..train_len));
        }

        let started = Instant::now();
        let stats = train_step(
            &file,
            &mut head,
            &mnist,
            &indices,
            cfg.layers,
            cfg.chunk_layers,
            cfg.alpha,
            cfg.layer_lr,
            cfg.head_lr,
        )?;
        head.save(&cfg.head)?;

        let elapsed = started.elapsed();
        let layer_samples = cfg.layers as f64 * cfg.batch as f64 * 3.0;
        let throughput = layer_samples / elapsed.as_secs_f64();
        println!(
            "step={} loss={:.6} batch_acc={:.3} elapsed={:.2?} throughput={:.2}M layer-samples/s",
            step + 1,
            stats.loss,
            stats.correct as f32 / cfg.batch as f32,
            elapsed,
            throughput / 1.0e6
        );
    }

    file.sync_data()?;
    Ok(())
}

fn eval(cfg: EvalConfig) -> Result<()> {
    if cfg.batch == 0 {
        return Err("--batch must be at least 1".into());
    }
    if cfg.chunk_layers == 0 {
        return Err("--chunk-layers must be at least 1".into());
    }

    verify_weights_len(&cfg.weights, cfg.layers)?;
    let mnist = Mnist::load(&cfg.data_dir)?;
    let head = Head::load(&cfg.head)?;
    let file = OpenOptions::new().read(true).open(&cfg.weights)?;

    let samples = cfg.samples.min(mnist.test_len());
    let mut correct = 0usize;
    let mut total_loss = 0.0f32;
    let started = Instant::now();

    for start in (0..samples).step_by(cfg.batch) {
        let end = (start + cfg.batch).min(samples);
        let indices: Vec<_> = (start..end).collect();
        let mut h = input_forward(&head, &mnist.test_images, &indices);
        forward_all_layers(&file, &mut h, cfg.layers, cfg.chunk_layers, cfg.alpha)?;

        let (loss, batch_correct) = eval_output(&head, &h, &mnist.test_labels, &indices);
        total_loss += loss * indices.len() as f32;
        correct += batch_correct;
    }

    println!(
        "eval samples={} loss={:.6} acc={:.3} elapsed={:.2?}",
        samples,
        total_loss / samples as f32,
        correct as f32 / samples as f32,
        started.elapsed()
    );
    Ok(())
}

fn train_dense(cfg: DenseTrainConfig) -> Result<()> {
    validate_width(cfg.width)?;
    if cfg.batch == 0 {
        return Err("--batch must be at least 1".into());
    }
    if cfg.chunk_layers == 0 {
        return Err("--chunk-layers must be at least 1".into());
    }

    if !cfg.weights.exists() {
        if cfg.init_if_missing {
            init_dense_weights(
                &cfg.weights,
                cfg.layers,
                cfg.width,
                cfg.chunk_layers,
                cfg.seed,
                0.001,
            )?;
        } else {
            return Err(format!("{} does not exist", cfg.weights.display()).into());
        }
    }

    verify_dense_weights_len(&cfg.weights, cfg.layers, cfg.width)?;

    let mnist = Mnist::load(&cfg.data_dir)?;
    let mut head = DenseHead::load_or_init(&cfg.head, cfg.width, cfg.seed ^ 0xD15EA5E, 1.0)?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&cfg.weights)?;

    let train_len = cfg.train_limit.min(mnist.train_len());
    if train_len == 0 {
        return Err("no MNIST training samples selected".into());
    }

    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    println!(
        "train-dense layers={} width={} alpha={} batch={} chunk_layers={} train_len={} weight_file={:.3} GB",
        cfg.layers,
        cfg.width,
        cfg.alpha,
        cfg.batch,
        cfg.chunk_layers,
        train_len,
        dense_total_bytes(cfg.layers, cfg.width)? as f64 / 1.0e9
    );

    for step in 0..cfg.steps {
        let mut indices = Vec::with_capacity(cfg.batch);
        for _ in 0..cfg.batch {
            indices.push(rng.gen_range(0..train_len));
        }

        let started = Instant::now();
        let stats = dense_train_step(
            &file,
            &mut head,
            &mnist,
            &indices,
            cfg.layers,
            cfg.chunk_layers,
            cfg.alpha,
            cfg.layer_lr,
            cfg.head_lr,
        )?;
        head.save(&cfg.head)?;

        let elapsed = started.elapsed();
        let layer_samples = cfg.layers as f64 * cfg.batch as f64 * 3.0;
        let throughput = layer_samples / elapsed.as_secs_f64();
        let should_report = step == 0
            || step + 1 == cfg.steps
            || (cfg.report_every != 0 && (step + 1) % cfg.report_every == 0);
        if should_report {
            println!(
                "step={} loss={:.6} batch_acc={:.3} elapsed={:.2?} throughput={:.2}M layer-samples/s",
                step + 1,
                stats.loss,
                stats.correct as f32 / cfg.batch as f32,
                elapsed,
                throughput / 1.0e6
            );
        }
    }

    file.sync_data()?;
    Ok(())
}

fn eval_dense(cfg: DenseEvalConfig) -> Result<()> {
    validate_width(cfg.width)?;
    if cfg.batch == 0 {
        return Err("--batch must be at least 1".into());
    }
    if cfg.chunk_layers == 0 {
        return Err("--chunk-layers must be at least 1".into());
    }

    verify_dense_weights_len(&cfg.weights, cfg.layers, cfg.width)?;
    let mnist = Mnist::load(&cfg.data_dir)?;
    let head = DenseHead::load(&cfg.head, cfg.width)?;
    let file = OpenOptions::new().read(true).open(&cfg.weights)?;

    let samples = cfg.samples.min(mnist.test_len());
    if samples == 0 {
        return Err("no MNIST eval samples selected".into());
    }

    let mut correct = 0usize;
    let mut total_loss = 0.0f32;
    let started = Instant::now();

    for start in (0..samples).step_by(cfg.batch) {
        let end = (start + cfg.batch).min(samples);
        let indices: Vec<_> = (start..end).collect();
        let mut h = dense_input_forward(&head, &mnist.test_images, &indices);
        dense_forward_all_layers(
            &file,
            cfg.width,
            &mut h,
            cfg.layers,
            cfg.chunk_layers,
            cfg.alpha,
        )?;

        let stats = dense_eval_output(&head, &h, &mnist.test_labels, &indices);
        total_loss += stats.loss * indices.len() as f32;
        correct += stats.correct;
    }

    println!(
        "eval-dense samples={} loss={:.6} acc={:.3} elapsed={:.2?}",
        samples,
        total_loss / samples as f32,
        correct as f32 / samples as f32,
        started.elapsed()
    );
    Ok(())
}

fn train_mlp(cfg: MlpTrainConfig) -> Result<()> {
    if cfg.hidden == 0 {
        return Err("--hidden must be at least 1".into());
    }
    if cfg.batch == 0 {
        return Err("--batch must be at least 1".into());
    }

    let mnist = Mnist::load(&cfg.data_dir)?;
    let train_len = cfg.train_limit.min(mnist.train_len());
    if train_len == 0 {
        return Err("no MNIST training samples selected".into());
    }

    let mut head = MlpHead::load_or_init(&cfg.head, cfg.hidden, cfg.seed ^ 0x51DECAFE)?;
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut indices = vec![0usize; cfg.batch];
    let started = Instant::now();

    println!(
        "train-mlp hidden={} steps={} batch={} lr={} train_len={}",
        cfg.hidden, cfg.steps, cfg.batch, cfg.lr, train_len
    );

    for step in 0..cfg.steps {
        for index in &mut indices {
            *index = rng.gen_range(0..train_len);
        }

        let stats = mlp_train_step(
            &mut head,
            &mnist.train_images,
            &mnist.train_labels,
            &indices,
            cfg.lr,
        );
        let should_report = step == 0
            || step + 1 == cfg.steps
            || (cfg.report_every != 0 && (step + 1) % cfg.report_every == 0);
        if should_report {
            println!(
                "step={} loss={:.6} batch_acc={:.3} elapsed={:.2?}",
                step + 1,
                stats.loss,
                stats.correct as f32 / cfg.batch as f32,
                started.elapsed()
            );
        }
    }

    head.save(&cfg.head)?;
    Ok(())
}

fn eval_mlp(cfg: MlpEvalConfig) -> Result<()> {
    if cfg.hidden == 0 {
        return Err("--hidden must be at least 1".into());
    }
    if cfg.batch == 0 {
        return Err("--batch must be at least 1".into());
    }

    let mnist = Mnist::load(&cfg.data_dir)?;
    let head = MlpHead::load(&cfg.head, cfg.hidden)?;
    let samples = cfg.samples.min(mnist.test_len());
    if samples == 0 {
        return Err("no MNIST eval samples selected".into());
    }

    let started = Instant::now();
    let mut total_loss = 0.0;
    let mut total_correct = 0usize;

    for start in (0..samples).step_by(cfg.batch) {
        let end = (start + cfg.batch).min(samples);
        let indices: Vec<_> = (start..end).collect();
        let stats = mlp_eval_batch(&head, &mnist.test_images, &mnist.test_labels, &indices);
        total_loss += stats.loss * indices.len() as f32;
        total_correct += stats.correct;
    }

    println!(
        "eval-mlp samples={} loss={:.6} acc={:.3} elapsed={:.2?}",
        samples,
        total_loss / samples as f32,
        total_correct as f32 / samples as f32,
        started.elapsed()
    );
    Ok(())
}

struct StepStats {
    loss: f32,
    correct: usize,
}

fn dense_train_step(
    file: &File,
    head: &mut DenseHead,
    mnist: &Mnist,
    indices: &[usize],
    layers: u64,
    chunk_layers: usize,
    alpha: f32,
    layer_lr: f32,
    head_lr: f32,
) -> Result<StepStats> {
    let width = head.width;
    let batch = indices.len();
    let chunks = div_ceil(layers, chunk_layers as u64) as usize;
    let mut h = dense_input_forward(head, &mnist.train_images, indices);
    let mut checkpoints = Vec::with_capacity(chunks + 1);
    checkpoints.push(h.clone());

    for chunk_idx in 0..chunks {
        let (start_layer, count) = chunk_range(chunk_idx, layers, chunk_layers);
        let params = read_dense_chunk(file, start_layer, count, width)?;
        dense_forward_chunk(&params, width, &mut h, alpha);
        checkpoints.push(h.clone());
    }

    let (stats, mut grad_h) =
        dense_output_loss_backward_update(head, &h, &mnist.train_labels, indices, head_lr);

    for chunk_idx in (0..chunks).rev() {
        let (start_layer, count) = chunk_range(chunk_idx, layers, chunk_layers);
        let mut params = read_dense_chunk(file, start_layer, count, width)?;
        grad_h = dense_backward_chunk(
            &mut params,
            width,
            &checkpoints[chunk_idx],
            &grad_h,
            batch,
            alpha,
            layer_lr,
        );
        write_dense_chunk(file, start_layer, width, &params)?;
    }

    dense_input_backward_update(
        head,
        &mnist.train_images,
        indices,
        &checkpoints[0],
        &grad_h,
        head_lr,
    );

    Ok(stats)
}

fn dense_forward_all_layers(
    file: &File,
    width: usize,
    h: &mut [f32],
    layers: u64,
    chunk_layers: usize,
    alpha: f32,
) -> Result<()> {
    let chunks = div_ceil(layers, chunk_layers as u64) as usize;
    for chunk_idx in 0..chunks {
        let (start_layer, count) = chunk_range(chunk_idx, layers, chunk_layers);
        let params = read_dense_chunk(file, start_layer, count, width)?;
        dense_forward_chunk(&params, width, h, alpha);
    }
    Ok(())
}

fn dense_input_forward(head: &DenseHead, images: &[f32], indices: &[usize]) -> Vec<f32> {
    let width = head.width;
    let mut h = vec![0.0; indices.len() * width];

    for (batch_idx, &image_idx) in indices.iter().enumerate() {
        let out_offset = batch_idx * width;
        h[out_offset..out_offset + width].copy_from_slice(&head.input_b);
        let image = &images[image_idx * INPUTS..(image_idx + 1) * INPUTS];

        for (pixel_idx, &x) in image.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            let weight_offset = pixel_idx * width;
            for width_idx in 0..width {
                h[out_offset + width_idx] += x * head.input_w[weight_offset + width_idx];
            }
        }

        for width_idx in 0..width {
            h[out_offset + width_idx] = h[out_offset + width_idx].tanh();
        }
    }

    h
}

fn dense_input_backward_update(
    head: &mut DenseHead,
    images: &[f32],
    indices: &[usize],
    input_h: &[f32],
    grad_h: &[f32],
    lr: f32,
) {
    let width = head.width;
    let mut grad_w = vec![0.0; INPUTS * width];
    let mut grad_b = vec![0.0; width];

    for (batch_idx, &image_idx) in indices.iter().enumerate() {
        let offset = batch_idx * width;
        let image = &images[image_idx * INPUTS..(image_idx + 1) * INPUTS];
        let mut grad_z = vec![0.0; width];

        for width_idx in 0..width {
            let h = input_h[offset + width_idx];
            grad_z[width_idx] = grad_h[offset + width_idx] * (1.0 - h * h);
            grad_b[width_idx] += grad_z[width_idx];
        }

        for (pixel_idx, &x) in image.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            let weight_offset = pixel_idx * width;
            for width_idx in 0..width {
                grad_w[weight_offset + width_idx] += x * grad_z[width_idx];
            }
        }
    }

    for (w, grad) in head.input_w.iter_mut().zip(grad_w) {
        *w -= lr * grad;
    }
    for (b, grad) in head.input_b.iter_mut().zip(grad_b) {
        *b -= lr * grad;
    }
}

fn dense_forward_chunk(params: &[f32], width: usize, h: &mut [f32], alpha: f32) {
    let layer_floats = dense_layer_floats(width);
    let batch = h.len() / width;
    let mut next = vec![0.0; width];

    for layer in params.chunks_exact(layer_floats) {
        let bias_offset = width * width;
        for sample_idx in 0..batch {
            let sample_offset = sample_idx * width;
            for out_idx in 0..width {
                let mut sum = layer[bias_offset + out_idx];
                for in_idx in 0..width {
                    sum += h[sample_offset + in_idx] * layer[in_idx * width + out_idx];
                }
                next[out_idx] = h[sample_offset + out_idx] + alpha * sum.tanh();
            }
            h[sample_offset..sample_offset + width].copy_from_slice(&next);
        }
    }
}

fn dense_backward_chunk(
    params: &mut [f32],
    width: usize,
    start_h: &[f32],
    end_grad: &[f32],
    batch: usize,
    alpha: f32,
    lr: f32,
) -> Vec<f32> {
    let layer_floats = dense_layer_floats(width);
    let layers = params.len() / layer_floats;
    let stride = batch * width;
    let mut activations = vec![0.0; (layers + 1) * stride];
    activations[..stride].copy_from_slice(start_h);

    for layer_idx in 0..layers {
        let layer_offset = layer_idx * layer_floats;
        let prev_offset = layer_idx * stride;
        let next_offset = (layer_idx + 1) * stride;
        let layer = &params[layer_offset..layer_offset + layer_floats];
        let bias_offset = width * width;

        for sample_idx in 0..batch {
            let prev_sample = prev_offset + sample_idx * width;
            let next_sample = next_offset + sample_idx * width;
            for out_idx in 0..width {
                let mut sum = layer[bias_offset + out_idx];
                for in_idx in 0..width {
                    sum += activations[prev_sample + in_idx] * layer[in_idx * width + out_idx];
                }
                activations[next_sample + out_idx] =
                    activations[prev_sample + out_idx] + alpha * sum.tanh();
            }
        }
    }

    let mut grad = end_grad.to_vec();
    let mut prev_grad = vec![0.0; stride];
    let mut du = vec![0.0; width];
    let mut grad_params = vec![0.0; layer_floats];
    let bias_offset = width * width;

    for layer_idx in (0..layers).rev() {
        let layer_offset = layer_idx * layer_floats;
        let act_offset = layer_idx * stride;
        grad_params.fill(0.0);
        prev_grad.fill(0.0);

        for sample_idx in 0..batch {
            let sample_offset = sample_idx * width;
            let h_offset = act_offset + sample_offset;

            for out_idx in 0..width {
                let mut sum = params[layer_offset + bias_offset + out_idx];
                for in_idx in 0..width {
                    sum += activations[h_offset + in_idx]
                        * params[layer_offset + in_idx * width + out_idx];
                }
                let a = sum.tanh();
                du[out_idx] = grad[sample_offset + out_idx] * alpha * (1.0 - a * a);
                grad_params[bias_offset + out_idx] += du[out_idx];
            }

            for in_idx in 0..width {
                let h_value = activations[h_offset + in_idx];
                let mut g = grad[sample_offset + in_idx];
                for out_idx in 0..width {
                    grad_params[in_idx * width + out_idx] += h_value * du[out_idx];
                    g += du[out_idx] * params[layer_offset + in_idx * width + out_idx];
                }
                prev_grad[sample_offset + in_idx] = g;
            }
        }

        for param_idx in 0..layer_floats {
            params[layer_offset + param_idx] -= lr * grad_params[param_idx];
        }

        std::mem::swap(&mut grad, &mut prev_grad);
    }

    grad
}

fn dense_output_loss_backward_update(
    head: &mut DenseHead,
    final_h: &[f32],
    labels: &[u8],
    indices: &[usize],
    lr: f32,
) -> (StepStats, Vec<f32>) {
    let width = head.width;
    let batch = indices.len();
    let inv_batch = 1.0 / batch as f32;
    let old_w = head.output_w.clone();
    let mut grad_h = vec![0.0; batch * width];
    let mut grad_w = vec![0.0; width * CLASSES];
    let mut grad_b = [0.0; CLASSES];
    let mut total_loss = 0.0;
    let mut correct = 0usize;

    for (batch_idx, &image_idx) in indices.iter().enumerate() {
        let sample_offset = batch_idx * width;
        let mut logits = head.output_b;

        for width_idx in 0..width {
            let h = final_h[sample_offset + width_idx];
            let weight_offset = width_idx * CLASSES;
            for class_idx in 0..CLASSES {
                logits[class_idx] += h * old_w[weight_offset + class_idx];
            }
        }

        let label = labels[image_idx] as usize;
        let (loss, predicted, grad_logits) = softmax_loss_gradient(logits, label, inv_batch);
        total_loss += loss;
        if predicted == label {
            correct += 1;
        }

        for class_idx in 0..CLASSES {
            grad_b[class_idx] += grad_logits[class_idx];
        }
        for width_idx in 0..width {
            let weight_offset = width_idx * CLASSES;
            let h = final_h[sample_offset + width_idx];
            for class_idx in 0..CLASSES {
                let grad_logit = grad_logits[class_idx];
                grad_w[weight_offset + class_idx] += h * grad_logit;
                grad_h[sample_offset + width_idx] += old_w[weight_offset + class_idx] * grad_logit;
            }
        }
    }

    for (w, grad) in head.output_w.iter_mut().zip(grad_w) {
        *w -= lr * grad;
    }
    for (b, grad) in head.output_b.iter_mut().zip(grad_b) {
        *b -= lr * grad;
    }

    (
        StepStats {
            loss: total_loss * inv_batch,
            correct,
        },
        grad_h,
    )
}

fn dense_eval_output(
    head: &DenseHead,
    final_h: &[f32],
    labels: &[u8],
    indices: &[usize],
) -> StepStats {
    let width = head.width;
    let mut total_loss = 0.0;
    let mut correct = 0usize;

    for (batch_idx, &image_idx) in indices.iter().enumerate() {
        let sample_offset = batch_idx * width;
        let mut logits = head.output_b;

        for width_idx in 0..width {
            let h = final_h[sample_offset + width_idx];
            let weight_offset = width_idx * CLASSES;
            for class_idx in 0..CLASSES {
                logits[class_idx] += h * head.output_w[weight_offset + class_idx];
            }
        }

        let label = labels[image_idx] as usize;
        let (loss, predicted, _) = softmax_loss_gradient(logits, label, 1.0);
        total_loss += loss;
        if predicted == label {
            correct += 1;
        }
    }

    StepStats {
        loss: total_loss / indices.len() as f32,
        correct,
    }
}

fn mlp_train_step(
    head: &mut MlpHead,
    images: &[f32],
    labels: &[u8],
    indices: &[usize],
    lr: f32,
) -> StepStats {
    let hidden = head.hidden;
    let batch = indices.len();
    let inv_batch = 1.0 / batch as f32;
    let old_w2 = head.w2.clone();

    let mut hidden_acts = vec![0.0f32; batch * hidden];
    let mut grad_w1 = vec![0.0f32; INPUTS * hidden];
    let mut grad_b1 = vec![0.0f32; hidden];
    let mut grad_w2 = vec![0.0f32; hidden * CLASSES];
    let mut grad_b2 = [0.0f32; CLASSES];

    let mut total_loss = 0.0;
    let mut correct = 0usize;

    for (batch_idx, &image_idx) in indices.iter().enumerate() {
        let image = &images[image_idx * INPUTS..(image_idx + 1) * INPUTS];
        let hidden_start = batch_idx * hidden;

        hidden_acts[hidden_start..hidden_start + hidden].copy_from_slice(&head.b1);
        for (pixel_idx, &x) in image.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            let weight_start = pixel_idx * hidden;
            for hidden_idx in 0..hidden {
                hidden_acts[hidden_start + hidden_idx] += x * head.w1[weight_start + hidden_idx];
            }
        }
        for hidden_idx in 0..hidden {
            hidden_acts[hidden_start + hidden_idx] =
                hidden_acts[hidden_start + hidden_idx].max(0.0);
        }

        let mut logits = head.b2;
        for hidden_idx in 0..hidden {
            let activation = hidden_acts[hidden_start + hidden_idx];
            if activation == 0.0 {
                continue;
            }
            let weight_start = hidden_idx * CLASSES;
            for class_idx in 0..CLASSES {
                logits[class_idx] += activation * old_w2[weight_start + class_idx];
            }
        }

        let label = labels[image_idx] as usize;
        let (loss, predicted, grad_logits) = softmax_loss_gradient(logits, label, inv_batch);
        total_loss += loss;
        if predicted == label {
            correct += 1;
        }

        let mut grad_hidden = vec![0.0f32; hidden];
        for class_idx in 0..CLASSES {
            grad_b2[class_idx] += grad_logits[class_idx];
        }
        for hidden_idx in 0..hidden {
            let activation = hidden_acts[hidden_start + hidden_idx];
            let weight_start = hidden_idx * CLASSES;
            for class_idx in 0..CLASSES {
                let grad_logit = grad_logits[class_idx];
                grad_w2[weight_start + class_idx] += activation * grad_logit;
                grad_hidden[hidden_idx] += old_w2[weight_start + class_idx] * grad_logit;
            }
        }

        for hidden_idx in 0..hidden {
            if hidden_acts[hidden_start + hidden_idx] <= 0.0 {
                grad_hidden[hidden_idx] = 0.0;
            }
            grad_b1[hidden_idx] += grad_hidden[hidden_idx];
        }

        for (pixel_idx, &x) in image.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            let weight_start = pixel_idx * hidden;
            for hidden_idx in 0..hidden {
                grad_w1[weight_start + hidden_idx] += x * grad_hidden[hidden_idx];
            }
        }
    }

    for (w, grad) in head.w1.iter_mut().zip(grad_w1) {
        *w -= lr * grad;
    }
    for (b, grad) in head.b1.iter_mut().zip(grad_b1) {
        *b -= lr * grad;
    }
    for (w, grad) in head.w2.iter_mut().zip(grad_w2) {
        *w -= lr * grad;
    }
    for (b, grad) in head.b2.iter_mut().zip(grad_b2) {
        *b -= lr * grad;
    }

    StepStats {
        loss: total_loss * inv_batch,
        correct,
    }
}

fn mlp_eval_batch(head: &MlpHead, images: &[f32], labels: &[u8], indices: &[usize]) -> StepStats {
    let hidden = head.hidden;
    let mut total_loss = 0.0;
    let mut correct = 0usize;
    let mut activations = vec![0.0f32; hidden];

    for &image_idx in indices {
        let image = &images[image_idx * INPUTS..(image_idx + 1) * INPUTS];

        activations.copy_from_slice(&head.b1);
        for (pixel_idx, &x) in image.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            let weight_start = pixel_idx * hidden;
            for (hidden_idx, activation) in activations.iter_mut().enumerate() {
                *activation += x * head.w1[weight_start + hidden_idx];
            }
        }
        for activation in &mut activations {
            *activation = activation.max(0.0);
        }

        let mut logits = head.b2;
        for (hidden_idx, &activation) in activations.iter().enumerate() {
            if activation == 0.0 {
                continue;
            }
            let weight_start = hidden_idx * CLASSES;
            for (class_idx, logit) in logits.iter_mut().enumerate() {
                *logit += activation * head.w2[weight_start + class_idx];
            }
        }

        let label = labels[image_idx] as usize;
        let (loss, predicted, _) = softmax_loss_gradient(logits, label, 1.0);
        total_loss += loss;
        if predicted == label {
            correct += 1;
        }
    }

    StepStats {
        loss: total_loss / indices.len() as f32,
        correct,
    }
}

fn softmax_loss_gradient(
    mut logits: [f32; CLASSES],
    label: usize,
    gradient_scale: f32,
) -> (f32, usize, [f32; CLASSES]) {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0;
    for logit in &mut logits {
        *logit = (*logit - max_logit).exp();
        exp_sum += *logit;
    }

    let mut loss = 0.0;
    let mut predicted = 0usize;
    let mut predicted_prob = -1.0f32;
    let mut grad = [0.0; CLASSES];

    for class_idx in 0..CLASSES {
        let prob = logits[class_idx] / exp_sum;
        if prob > predicted_prob {
            predicted_prob = prob;
            predicted = class_idx;
        }

        let mut dlogit = prob;
        if class_idx == label {
            loss = -prob.max(1.0e-20).ln();
            dlogit -= 1.0;
        }
        grad[class_idx] = dlogit * gradient_scale;
    }

    (loss, predicted, grad)
}

fn train_step(
    file: &File,
    head: &mut Head,
    mnist: &Mnist,
    indices: &[usize],
    layers: u64,
    chunk_layers: usize,
    alpha: f32,
    layer_lr: f32,
    head_lr: f32,
) -> Result<StepStats> {
    let batch = indices.len();
    let chunks = div_ceil(layers, chunk_layers as u64) as usize;
    let mut h = input_forward(head, &mnist.train_images, indices);
    let mut checkpoints = Vec::with_capacity(chunks + 1);
    checkpoints.push(h.clone());

    for chunk_idx in 0..chunks {
        let (start_layer, count) = chunk_range(chunk_idx, layers, chunk_layers);
        let params = read_chunk(file, start_layer, count)?;
        forward_chunk(&params, &mut h, alpha);
        checkpoints.push(h.clone());
    }

    let (loss, correct, mut grad_h) =
        output_loss_backward_update(head, &h, &mnist.train_labels, indices, head_lr);

    for chunk_idx in (0..chunks).rev() {
        let (start_layer, count) = chunk_range(chunk_idx, layers, chunk_layers);
        let mut params = read_chunk(file, start_layer, count)?;
        grad_h = backward_chunk(
            &mut params,
            &checkpoints[chunk_idx],
            &grad_h,
            batch,
            alpha,
            layer_lr,
        );
        write_chunk(file, start_layer, &params)?;
    }

    input_backward_update(
        head,
        &mnist.train_images,
        indices,
        &checkpoints[0],
        &grad_h,
        head_lr,
    );

    Ok(StepStats { loss, correct })
}

fn forward_all_layers(
    file: &File,
    h: &mut [[f32; WIDTH]],
    layers: u64,
    chunk_layers: usize,
    alpha: f32,
) -> Result<()> {
    let chunks = div_ceil(layers, chunk_layers as u64) as usize;
    for chunk_idx in 0..chunks {
        let (start_layer, count) = chunk_range(chunk_idx, layers, chunk_layers);
        let params = read_chunk(file, start_layer, count)?;
        forward_chunk(&params, h, alpha);
    }
    Ok(())
}

fn input_forward(head: &Head, images: &[f32], indices: &[usize]) -> Vec<[f32; WIDTH]> {
    let mut h = vec![[0.0; WIDTH]; indices.len()];

    for (batch_idx, &image_idx) in indices.iter().enumerate() {
        let image = &images[image_idx * INPUTS..(image_idx + 1) * INPUTS];
        let mut h0 = head.input_b[0];
        let mut h1 = head.input_b[1];

        for (pixel_idx, &x) in image.iter().enumerate() {
            let weight_idx = pixel_idx * WIDTH;
            h0 += x * head.input_w[weight_idx];
            h1 += x * head.input_w[weight_idx + 1];
        }

        h[batch_idx] = [h0.tanh(), h1.tanh()];
    }

    h
}

fn input_backward_update(
    head: &mut Head,
    images: &[f32],
    indices: &[usize],
    input_h: &[[f32; WIDTH]],
    grad_h: &[[f32; WIDTH]],
    lr: f32,
) {
    let mut grad_w = vec![0.0; INPUTS * WIDTH];
    let mut grad_b = [0.0; WIDTH];

    for (batch_idx, &image_idx) in indices.iter().enumerate() {
        let image = &images[image_idx * INPUTS..(image_idx + 1) * INPUTS];
        let d0 = grad_h[batch_idx][0] * (1.0 - input_h[batch_idx][0] * input_h[batch_idx][0]);
        let d1 = grad_h[batch_idx][1] * (1.0 - input_h[batch_idx][1] * input_h[batch_idx][1]);

        grad_b[0] += d0;
        grad_b[1] += d1;

        for (pixel_idx, &x) in image.iter().enumerate() {
            let weight_idx = pixel_idx * WIDTH;
            grad_w[weight_idx] += x * d0;
            grad_w[weight_idx + 1] += x * d1;
        }
    }

    for (w, grad) in head.input_w.iter_mut().zip(grad_w) {
        *w -= lr * grad;
    }
    head.input_b[0] -= lr * grad_b[0];
    head.input_b[1] -= lr * grad_b[1];
}

fn forward_chunk(params: &[f32], h: &mut [[f32; WIDTH]], alpha: f32) {
    for layer in params.chunks_exact(LAYER_FLOATS) {
        let w00 = layer[0];
        let w01 = layer[1];
        let w10 = layer[2];
        let w11 = layer[3];
        let b0 = layer[4];
        let b1 = layer[5];

        for sample in h.iter_mut() {
            let h0 = sample[0];
            let h1 = sample[1];
            let a0 = (h0 * w00 + h1 * w10 + b0).tanh();
            let a1 = (h0 * w01 + h1 * w11 + b1).tanh();
            sample[0] = h0 + alpha * a0;
            sample[1] = h1 + alpha * a1;
        }
    }
}

fn backward_chunk(
    params: &mut [f32],
    start_h: &[[f32; WIDTH]],
    end_grad: &[[f32; WIDTH]],
    batch: usize,
    alpha: f32,
    lr: f32,
) -> Vec<[f32; WIDTH]> {
    let layers = params.len() / LAYER_FLOATS;
    let mut activations = vec![[0.0; WIDTH]; (layers + 1) * batch];
    activations[..batch].copy_from_slice(start_h);

    for layer_idx in 0..layers {
        let prev_offset = layer_idx * batch;
        let next_offset = (layer_idx + 1) * batch;
        let layer = &params[layer_idx * LAYER_FLOATS..(layer_idx + 1) * LAYER_FLOATS];
        let w00 = layer[0];
        let w01 = layer[1];
        let w10 = layer[2];
        let w11 = layer[3];
        let b0 = layer[4];
        let b1 = layer[5];

        for sample_idx in 0..batch {
            let h0 = activations[prev_offset + sample_idx][0];
            let h1 = activations[prev_offset + sample_idx][1];
            let a0 = (h0 * w00 + h1 * w10 + b0).tanh();
            let a1 = (h0 * w01 + h1 * w11 + b1).tanh();
            activations[next_offset + sample_idx] = [h0 + alpha * a0, h1 + alpha * a1];
        }
    }

    let mut grad = end_grad.to_vec();
    let mut prev_grad = vec![[0.0; WIDTH]; batch];

    for layer_idx in (0..layers).rev() {
        let act_offset = layer_idx * batch;
        let param_offset = layer_idx * LAYER_FLOATS;

        let w00 = params[param_offset];
        let w01 = params[param_offset + 1];
        let w10 = params[param_offset + 2];
        let w11 = params[param_offset + 3];
        let b0 = params[param_offset + 4];
        let b1 = params[param_offset + 5];

        let mut gw00 = 0.0;
        let mut gw01 = 0.0;
        let mut gw10 = 0.0;
        let mut gw11 = 0.0;
        let mut gb0 = 0.0;
        let mut gb1 = 0.0;

        for sample_idx in 0..batch {
            let h0 = activations[act_offset + sample_idx][0];
            let h1 = activations[act_offset + sample_idx][1];
            let gy0 = grad[sample_idx][0];
            let gy1 = grad[sample_idx][1];

            let a0 = (h0 * w00 + h1 * w10 + b0).tanh();
            let a1 = (h0 * w01 + h1 * w11 + b1).tanh();
            let du0 = gy0 * alpha * (1.0 - a0 * a0);
            let du1 = gy1 * alpha * (1.0 - a1 * a1);

            gw00 += h0 * du0;
            gw10 += h1 * du0;
            gb0 += du0;

            gw01 += h0 * du1;
            gw11 += h1 * du1;
            gb1 += du1;

            prev_grad[sample_idx][0] = gy0 + du0 * w00 + du1 * w01;
            prev_grad[sample_idx][1] = gy1 + du0 * w10 + du1 * w11;
        }

        params[param_offset] -= lr * gw00;
        params[param_offset + 1] -= lr * gw01;
        params[param_offset + 2] -= lr * gw10;
        params[param_offset + 3] -= lr * gw11;
        params[param_offset + 4] -= lr * gb0;
        params[param_offset + 5] -= lr * gb1;

        std::mem::swap(&mut grad, &mut prev_grad);
    }

    grad
}

fn output_loss_backward_update(
    head: &mut Head,
    final_h: &[[f32; WIDTH]],
    labels: &[u8],
    indices: &[usize],
    lr: f32,
) -> (f32, usize, Vec<[f32; WIDTH]>) {
    let old_w = head.output_w;
    let old_b = head.output_b;
    let batch = indices.len();
    let inv_batch = 1.0 / batch as f32;

    let mut loss = 0.0;
    let mut correct = 0usize;
    let mut grad_h = vec![[0.0; WIDTH]; batch];
    let mut grad_w = [0.0; WIDTH * CLASSES];
    let mut grad_b = [0.0; CLASSES];

    for (batch_idx, &image_idx) in indices.iter().enumerate() {
        let mut logits = [0.0; CLASSES];
        for class_idx in 0..CLASSES {
            logits[class_idx] = old_b[class_idx]
                + final_h[batch_idx][0] * old_w[class_idx]
                + final_h[batch_idx][1] * old_w[CLASSES + class_idx];
        }

        let label = labels[image_idx] as usize;
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0;
        for logit in &mut logits {
            *logit = (*logit - max_logit).exp();
            exp_sum += *logit;
        }

        let mut pred = 0usize;
        let mut pred_prob = -1.0f32;
        for class_idx in 0..CLASSES {
            let prob = logits[class_idx] / exp_sum;
            if prob > pred_prob {
                pred_prob = prob;
                pred = class_idx;
            }

            let mut dlogit = prob;
            if class_idx == label {
                loss -= prob.max(1.0e-20).ln();
                dlogit -= 1.0;
            }
            dlogit *= inv_batch;

            grad_b[class_idx] += dlogit;
            grad_w[class_idx] += final_h[batch_idx][0] * dlogit;
            grad_w[CLASSES + class_idx] += final_h[batch_idx][1] * dlogit;
            grad_h[batch_idx][0] += dlogit * old_w[class_idx];
            grad_h[batch_idx][1] += dlogit * old_w[CLASSES + class_idx];
        }

        if pred == label {
            correct += 1;
        }
    }

    for class_idx in 0..CLASSES {
        head.output_b[class_idx] -= lr * grad_b[class_idx];
        head.output_w[class_idx] -= lr * grad_w[class_idx];
        head.output_w[CLASSES + class_idx] -= lr * grad_w[CLASSES + class_idx];
    }

    (loss * inv_batch, correct, grad_h)
}

fn eval_output(
    head: &Head,
    final_h: &[[f32; WIDTH]],
    labels: &[u8],
    indices: &[usize],
) -> (f32, usize) {
    let mut loss = 0.0;
    let mut correct = 0usize;

    for (batch_idx, &image_idx) in indices.iter().enumerate() {
        let mut logits = [0.0; CLASSES];
        for class_idx in 0..CLASSES {
            logits[class_idx] = head.output_b[class_idx]
                + final_h[batch_idx][0] * head.output_w[class_idx]
                + final_h[batch_idx][1] * head.output_w[CLASSES + class_idx];
        }

        let label = labels[image_idx] as usize;
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0;
        for logit in &mut logits {
            *logit = (*logit - max_logit).exp();
            exp_sum += *logit;
        }

        let mut pred = 0usize;
        let mut pred_prob = -1.0f32;
        for class_idx in 0..CLASSES {
            let prob = logits[class_idx] / exp_sum;
            if prob > pred_prob {
                pred_prob = prob;
                pred = class_idx;
            }
            if class_idx == label {
                loss -= prob.max(1.0e-20).ln();
            }
        }

        if pred == label {
            correct += 1;
        }
    }

    (loss / indices.len() as f32, correct)
}

fn init_weights(
    path: &Path,
    layers: u64,
    chunk_layers: usize,
    seed: u64,
    scale: f32,
) -> Result<()> {
    if chunk_layers == 0 {
        return Err("--chunk-layers must be at least 1".into());
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let total_bytes = layers
        .checked_mul(LAYER_BYTES)
        .ok_or("layer file would overflow u64 length")?;
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(path)?;
    file.set_len(total_bytes)?;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let chunks = div_ceil(layers, chunk_layers as u64) as usize;
    let started = Instant::now();

    println!(
        "init {} layers at {} ({:.3} GB)",
        layers,
        path.display(),
        total_bytes as f64 / 1.0e9
    );

    for chunk_idx in 0..chunks {
        let (start_layer, count) = chunk_range(chunk_idx, layers, chunk_layers);
        let mut params = vec![0.0f32; count * LAYER_FLOATS];
        for value in &mut params {
            *value = rng.gen_range(-scale..scale);
        }
        write_chunk(&file, start_layer, &params)?;

        if chunk_idx == 0 || chunk_idx + 1 == chunks || (chunk_idx + 1) % 10 == 0 {
            let done_layers = (start_layer + count as u64).min(layers);
            println!(
                "  wrote {}/{} layers ({:.1}%)",
                done_layers,
                layers,
                done_layers as f64 * 100.0 / layers as f64
            );
        }
    }

    file.sync_data()?;
    println!("done in {:.2?}", started.elapsed());
    Ok(())
}

fn verify_weights_len(path: &Path, layers: u64) -> Result<()> {
    let expected = layers
        .checked_mul(LAYER_BYTES)
        .ok_or("layer file would overflow u64 length")?;
    let actual = fs::metadata(path)?.len();
    if actual != expected {
        return Err(format!(
            "{} is {} bytes, expected {} for {} layers",
            path.display(),
            actual,
            expected,
            layers
        )
        .into());
    }
    Ok(())
}

fn read_chunk(file: &File, start_layer: u64, count: usize) -> Result<Vec<f32>> {
    let mut params = vec![0.0f32; count * LAYER_FLOATS];
    let offset = start_layer * LAYER_BYTES;
    read_exact_at_all(file, bytemuck::cast_slice_mut(&mut params), offset)?;
    Ok(params)
}

fn write_chunk(file: &File, start_layer: u64, params: &[f32]) -> Result<()> {
    let offset = start_layer * LAYER_BYTES;
    write_all_at_all(file, bytemuck::cast_slice(params), offset)?;
    Ok(())
}

fn init_dense_weights(
    path: &Path,
    layers: u64,
    width: usize,
    chunk_layers: usize,
    seed: u64,
    scale: f32,
) -> Result<()> {
    validate_width(width)?;
    if chunk_layers == 0 {
        return Err("--chunk-layers must be at least 1".into());
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let total_bytes = dense_total_bytes(layers, width)?;
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(path)?;
    file.set_len(total_bytes)?;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let chunks = div_ceil(layers, chunk_layers as u64) as usize;
    let layer_floats = dense_layer_floats(width);
    let started = Instant::now();

    println!(
        "init-dense {} layers width {} at {} ({:.3} GB)",
        layers,
        width,
        path.display(),
        total_bytes as f64 / 1.0e9
    );

    for chunk_idx in 0..chunks {
        let (start_layer, count) = chunk_range(chunk_idx, layers, chunk_layers);
        let mut params = vec![0.0f32; count * layer_floats];
        for layer in params.chunks_exact_mut(layer_floats) {
            for value in &mut layer[..width * width] {
                *value = rng.gen_range(-scale..scale);
            }
            for value in &mut layer[width * width..] {
                *value = 0.0;
            }
        }
        write_dense_chunk(&file, start_layer, width, &params)?;

        if chunk_idx == 0 || chunk_idx + 1 == chunks || (chunk_idx + 1) % 10 == 0 {
            let done_layers = (start_layer + count as u64).min(layers);
            println!(
                "  wrote {}/{} layers ({:.1}%)",
                done_layers,
                layers,
                done_layers as f64 * 100.0 / layers as f64
            );
        }
    }

    file.sync_data()?;
    println!("done in {:.2?}", started.elapsed());
    Ok(())
}

fn verify_dense_weights_len(path: &Path, layers: u64, width: usize) -> Result<()> {
    let expected = dense_total_bytes(layers, width)?;
    let actual = fs::metadata(path)?.len();
    if actual != expected {
        return Err(format!(
            "{} is {} bytes, expected {} for {} layers at width {}",
            path.display(),
            actual,
            expected,
            layers,
            width
        )
        .into());
    }
    Ok(())
}

fn read_dense_chunk(file: &File, start_layer: u64, count: usize, width: usize) -> Result<Vec<f32>> {
    let layer_floats = dense_layer_floats(width);
    let mut params = vec![0.0f32; count * layer_floats];
    let offset = dense_byte_offset(start_layer, width)?;
    read_exact_at_all(file, bytemuck::cast_slice_mut(&mut params), offset)?;
    Ok(params)
}

fn write_dense_chunk(file: &File, start_layer: u64, width: usize, params: &[f32]) -> Result<()> {
    let offset = dense_byte_offset(start_layer, width)?;
    write_all_at_all(file, bytemuck::cast_slice(params), offset)?;
    Ok(())
}

fn dense_layer_floats(width: usize) -> usize {
    width * width + width
}

fn dense_layer_bytes(width: usize) -> Result<u64> {
    let floats = dense_layer_floats(width);
    let bytes = floats
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or("dense layer byte count overflowed usize")?;
    Ok(bytes as u64)
}

fn dense_total_bytes(layers: u64, width: usize) -> Result<u64> {
    layers
        .checked_mul(dense_layer_bytes(width)?)
        .ok_or_else(|| "dense layer file would overflow u64 length".into())
}

fn dense_byte_offset(start_layer: u64, width: usize) -> Result<u64> {
    start_layer
        .checked_mul(dense_layer_bytes(width)?)
        .ok_or_else(|| "dense file offset would overflow u64".into())
}

fn validate_width(width: usize) -> Result<()> {
    if width == 0 {
        return Err("--width must be at least 1".into());
    }
    if width > 512 {
        return Err("--width above 512 is probably a mistake for this CPU trainer".into());
    }
    Ok(())
}

fn read_exact_at_all(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let bytes_read = file.read_at(&mut buf[done..], offset + done as u64)?;
        if bytes_read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short positioned read",
            ));
        }
        done += bytes_read;
    }
    Ok(())
}

fn write_all_at_all(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        let bytes_written = file.write_at(&buf[done..], offset + done as u64)?;
        if bytes_written == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "short positioned write",
            ));
        }
        done += bytes_written;
    }
    Ok(())
}

fn chunk_range(chunk_idx: usize, layers: u64, chunk_layers: usize) -> (u64, usize) {
    let start = chunk_idx as u64 * chunk_layers as u64;
    let remaining = layers - start;
    let count = remaining.min(chunk_layers as u64) as usize;
    (start, count)
}

fn div_ceil(x: u64, y: u64) -> u64 {
    x.div_ceil(y)
}

fn load_idx_file(data_dir: &Path, file_name: &str, url: &str) -> Result<Vec<u8>> {
    let path = data_dir.join(file_name);
    if path.exists() {
        return Ok(fs::read(path)?);
    }

    println!("download {}", url);
    let response = ureq::get(url).call()?;
    let mut decoder = GzDecoder::new(response.into_reader());
    let mut data = Vec::new();
    decoder.read_to_end(&mut data)?;
    fs::write(&path, &data)?;
    Ok(data)
}

fn parse_images(data: &[u8]) -> Result<(Vec<f32>, usize)> {
    if data.len() < 16 {
        return Err("IDX image file too short".into());
    }

    let magic = be_u32(data, 0);
    let count = be_u32(data, 4) as usize;
    let rows = be_u32(data, 8) as usize;
    let cols = be_u32(data, 12) as usize;

    if magic != 2051 {
        return Err(format!("bad image magic {}, expected 2051", magic).into());
    }
    if rows != 28 || cols != 28 {
        return Err(format!("bad MNIST image dimensions {}x{}", rows, cols).into());
    }

    let expected = 16 + count * INPUTS;
    if data.len() != expected {
        return Err(format!(
            "bad image file length {}, expected {}",
            data.len(),
            expected
        )
        .into());
    }

    let mut images = vec![0.0; count * INPUTS];
    for (dst, &src) in images.iter_mut().zip(&data[16..]) {
        *dst = src as f32 / 255.0;
    }

    Ok((images, count))
}

fn parse_labels(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 8 {
        return Err("IDX label file too short".into());
    }

    let magic = be_u32(data, 0);
    let count = be_u32(data, 4) as usize;

    if magic != 2049 {
        return Err(format!("bad label magic {}, expected 2049", magic).into());
    }

    let expected = 8 + count;
    if data.len() != expected {
        return Err(format!(
            "bad label file length {}, expected {}",
            data.len(),
            expected
        )
        .into());
    }

    Ok(data[8..].to_vec())
}

fn be_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap())
}
