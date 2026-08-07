use serde::Serialize;

const EPSILON: f32 = 1e-12;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum QuantizerKind {
    Sq4,
    Sq8,
    Lvq4,
    Lvq8,
}

impl QuantizerKind {
    pub(crate) const ALL: [Self; 4] = [Self::Sq4, Self::Sq8, Self::Lvq4, Self::Lvq8];

    pub(crate) const fn bits(self) -> u8 {
        match self {
            Self::Sq4 | Self::Lvq4 => 4,
            Self::Sq8 | Self::Lvq8 => 8,
        }
    }

    pub(crate) const fn is_lvq(self) -> bool {
        matches!(self, Self::Lvq4 | Self::Lvq8)
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Sq4 => "sq4",
            Self::Sq8 => "sq8",
            Self::Lvq4 => "lvq4",
            Self::Lvq8 => "lvq8",
        }
    }
}

pub(crate) struct Quantizer {
    pub(crate) kind: QuantizerKind,
    pub(crate) lower: Vec<f32>,
    pub(crate) scale: Vec<f32>,
}

pub(crate) struct EncodedVectors {
    pub(crate) codes: Vec<u8>,
    pub(crate) offsets: Option<Vec<f32>>,
    pub(crate) scales: Option<Vec<f32>>,
}

impl Quantizer {
    pub(crate) fn train(
        kind: QuantizerKind,
        vectors: &[f32],
        rows: usize,
        dimension: usize,
    ) -> Self {
        if kind.is_lvq() {
            return Self {
                kind,
                lower: Vec::new(),
                scale: Vec::new(),
            };
        }

        let mut lower = vec![f32::INFINITY; dimension];
        let mut upper = vec![f32::NEG_INFINITY; dimension];
        for row in vectors.chunks_exact(dimension).take(rows) {
            for (dim, &value) in row.iter().enumerate() {
                lower[dim] = lower[dim].min(value);
                upper[dim] = upper[dim].max(value);
            }
        }
        let levels = f32::from((1_u16 << kind.bits()) - 1);
        let scale = lower
            .iter()
            .zip(&upper)
            .map(|(&minimum, &maximum)| (maximum - minimum) / levels)
            .collect();
        Self { kind, lower, scale }
    }

    pub(crate) fn encode(&self, vectors: &[f32], rows: usize, dimension: usize) -> EncodedVectors {
        let levels = ((1_u16 << self.kind.bits()) - 1) as u8;
        let mut codes = vec![0_u8; rows * dimension];
        let mut offsets = self.kind.is_lvq().then(|| vec![0.0; rows]);
        let mut scales = self.kind.is_lvq().then(|| vec![0.0; rows]);

        for (row_index, row) in vectors.chunks_exact(dimension).take(rows).enumerate() {
            let (lower, scale) = if self.kind.is_lvq() {
                let lower = row.iter().copied().fold(f32::INFINITY, f32::min);
                let upper = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let scale = (upper - lower) / f32::from(levels);
                offsets.as_mut().expect("LVQ offsets")[row_index] = lower;
                scales.as_mut().expect("LVQ scales")[row_index] = scale;
                (None, None)
            } else {
                (Some(&self.lower[..]), Some(&self.scale[..]))
            };

            let row_codes = &mut codes[row_index * dimension..(row_index + 1) * dimension];
            for dim in 0..dimension {
                let offset = lower.map_or_else(
                    || offsets.as_ref().expect("LVQ offsets")[row_index],
                    |values| values[dim],
                );
                let width = scale.map_or_else(
                    || scales.as_ref().expect("LVQ scales")[row_index],
                    |values| values[dim],
                );
                row_codes[dim] = if width > EPSILON {
                    ((row[dim] - offset) / width)
                        .round()
                        .clamp(0.0, f32::from(levels)) as u8
                } else {
                    0
                };
            }
        }

        EncodedVectors {
            codes,
            offsets,
            scales,
        }
    }
}

pub(crate) fn pack_codes(codes: &[u8], rows: usize, dimension: usize, bits: u8) -> Vec<u8> {
    if bits == 8 {
        return codes.to_vec();
    }
    let stride = dimension.div_ceil(2);
    let mut packed = vec![0_u8; rows * stride];
    for row in 0..rows {
        for dim in 0..dimension {
            let code = codes[row * dimension + dim] & 0x0f;
            let slot = &mut packed[row * stride + dim / 2];
            if dim & 1 == 0 {
                *slot |= code;
            } else {
                *slot |= code << 4;
            }
        }
    }
    packed
}
