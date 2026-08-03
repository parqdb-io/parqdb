use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum BuildPhase {
    Pending = 0,
    ScanningSource = 1,
    ReadingTraining = 2,
    TrainingCentroids = 3,
    WritingCentroids = 4,
    BuildingPostings = 5,
    Publishing = 6,
    Complete = 7,
}

impl BuildPhase {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::ScanningSource,
            2 => Self::ReadingTraining,
            3 => Self::TrainingCentroids,
            4 => Self::WritingCentroids,
            5 => Self::BuildingPostings,
            6 => Self::Publishing,
            7 => Self::Complete,
            _ => Self::Pending,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::ScanningSource => "scanning_source",
            Self::ReadingTraining => "reading_training_vectors",
            Self::TrainingCentroids => "training_centroids",
            Self::WritingCentroids => "writing_centroids",
            Self::BuildingPostings => "building_postings",
            Self::Publishing => "publishing",
            Self::Complete => "complete",
        }
    }

    fn bounds(self) -> (f64, f64) {
        match self {
            Self::Pending => (0.0, 0.0),
            Self::ScanningSource => (0.0, 0.05),
            Self::ReadingTraining => (0.05, 0.15),
            Self::TrainingCentroids => (0.15, 0.65),
            Self::WritingCentroids => (0.65, 0.67),
            Self::BuildingPostings => (0.67, 0.98),
            Self::Publishing => (0.98, 1.0),
            Self::Complete => (1.0, 1.0),
        }
    }
}

#[derive(Debug, Default)]
struct ProgressState {
    phase: AtomicU8,
    completed: AtomicU64,
    total: AtomicU64,
}

/// Thread-safe progress state for one local index build.
#[derive(Debug, Clone, Default)]
pub struct LocalBuildProgress {
    state: Arc<ProgressState>,
}

/// One consistent-enough snapshot of local index-build progress.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalBuildProgressSnapshot {
    /// Current build phase.
    pub phase: &'static str,
    /// Work units completed in the current phase.
    pub completed: u64,
    /// Total work units in the current phase, or zero when unavailable.
    pub total: u64,
    /// Estimated overall completion in the inclusive range `0.0..=1.0`.
    pub fraction: f64,
}

impl LocalBuildProgress {
    pub(crate) fn begin(&self, phase: BuildPhase, total: usize) {
        self.state.total.store(as_u64(total), Ordering::Relaxed);
        self.state.completed.store(0, Ordering::Relaxed);
        self.state.phase.store(phase as u8, Ordering::Release);
    }

    pub(crate) fn set_completed(&self, completed: usize) {
        self.state
            .completed
            .store(as_u64(completed), Ordering::Relaxed);
    }

    pub(crate) fn advance(&self, amount: usize) -> u64 {
        let amount = as_u64(amount);
        let previous = self
            .state
            .completed
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |completed| {
                Some(completed.saturating_add(amount))
            })
            .unwrap_or_else(|completed| completed);
        previous.saturating_add(amount)
    }

    pub(crate) fn finish(&self) {
        self.begin(BuildPhase::Complete, 1);
        self.set_completed(1);
    }

    /// Returns the latest build phase and counters.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn snapshot(&self) -> LocalBuildProgressSnapshot {
        let phase = BuildPhase::from_u8(self.state.phase.load(Ordering::Acquire));
        let total = self.state.total.load(Ordering::Relaxed);
        let completed = self.state.completed.load(Ordering::Relaxed).min(total);
        let phase_fraction = if total == 0 {
            0.0
        } else {
            completed as f64 / total as f64
        };
        let (start, end) = phase.bounds();
        LocalBuildProgressSnapshot {
            phase: phase.name(),
            completed,
            total,
            fraction: start + (end - start) * phase_fraction,
        }
    }
}

fn as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{BuildPhase, LocalBuildProgress};

    #[test]
    fn progress_reports_phase_counters_and_overall_fraction() {
        let progress = LocalBuildProgress::default();
        progress.begin(BuildPhase::TrainingCentroids, 20);
        progress.set_completed(5);

        let snapshot = progress.snapshot();
        assert_eq!(snapshot.phase, "training_centroids");
        assert_eq!(snapshot.completed, 5);
        assert_eq!(snapshot.total, 20);
        assert!((snapshot.fraction - 0.275).abs() < f64::EPSILON);

        progress.finish();
        assert!((progress.snapshot().fraction - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn concurrent_progress_is_clamped_to_the_phase_total() {
        let progress = LocalBuildProgress::default();
        progress.begin(BuildPhase::ReadingTraining, 10);
        let _ = progress.advance(12);

        let snapshot = progress.snapshot();
        assert_eq!(snapshot.completed, 10);
        assert_eq!(snapshot.total, 10);
        assert!((snapshot.fraction - 0.15).abs() < f64::EPSILON);
    }
}
