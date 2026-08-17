#[cfg(target_os = "linux")]
use std::path::{Path, PathBuf};

const DEFAULT_MANAGED_MEMORY_NUMERATOR: usize = 4;
const DEFAULT_MANAGED_MEMORY_DENOMINATOR: usize = 5;
const DEFAULT_PAGE_CACHE_NUMERATOR: usize = 1;
const DEFAULT_PAGE_CACHE_DENOMINATOR: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AutomaticMemoryBudget {
    pub(super) execution: usize,
    pub(super) page_cache: usize,
}

pub(super) fn automatic_memory_budget(
    page_cache_capacity: Option<usize>,
) -> crate::Result<Option<AutomaticMemoryBudget>> {
    let Some(effective_limit) = effective_memory_limit() else {
        return Ok(None);
    };
    budget_for_limit(effective_limit, page_cache_capacity)
        .map(Some)
        .ok_or_else(|| {
            crate::Error::InvalidArgument(
                "relify.parquet.page_cache.capacity must be smaller than the automatic 80% memory budget"
                    .into(),
            )
        })
}

fn budget_for_limit(
    effective_limit: usize,
    page_cache_capacity: Option<usize>,
) -> Option<AutomaticMemoryBudget> {
    let managed = effective_limit.saturating_mul(DEFAULT_MANAGED_MEMORY_NUMERATOR)
        / DEFAULT_MANAGED_MEMORY_DENOMINATOR;
    let page_cache = page_cache_capacity.unwrap_or(
        managed.saturating_mul(DEFAULT_PAGE_CACHE_NUMERATOR) / DEFAULT_PAGE_CACHE_DENOMINATOR,
    );
    let execution = managed.checked_sub(page_cache)?;
    (execution > 0).then_some(AutomaticMemoryBudget {
        execution,
        page_cache,
    })
}

#[cfg(target_os = "linux")]
pub(crate) fn effective_memory_limit() -> Option<usize> {
    let cgroup_limit = cgroup_memory_limits().into_iter().min();
    let physical_memory = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                let value = line.strip_prefix("MemTotal:")?;
                let kib = value.split_whitespace().next()?.parse::<usize>().ok()?;
                kib.checked_mul(1024)
            })
        });
    cgroup_limit.into_iter().chain(physical_memory).min()
}

#[cfg(target_os = "linux")]
fn cgroup_memory_limits() -> Vec<usize> {
    fn collect_ancestors(start: PathBuf, root: &Path, file: &str, limits: &mut Vec<usize>) {
        let mut current = start;
        while current.starts_with(root) {
            limits.extend(read_memory_limit(&current.join(file)));
            if current == root || !current.pop() {
                break;
            }
        }
    }

    let mut limits = Vec::new();
    let Ok(contents) = std::fs::read_to_string("/proc/self/cgroup") else {
        return limits;
    };
    for line in contents.lines() {
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields.next().unwrap_or_default();
        let controllers = fields.next().unwrap_or_default();
        let relative = fields.next().unwrap_or_default().trim_start_matches('/');
        if hierarchy == "0" && controllers.is_empty() {
            let root = Path::new("/sys/fs/cgroup");
            collect_ancestors(root.join(relative), root, "memory.max", &mut limits);
        } else if controllers
            .split(',')
            .any(|controller| controller == "memory")
        {
            let root = Path::new("/sys/fs/cgroup/memory");
            collect_ancestors(
                root.join(relative),
                root,
                "memory.limit_in_bytes",
                &mut limits,
            );
        }
    }
    limits
}

#[cfg(target_os = "linux")]
fn read_memory_limit(path: &Path) -> Option<usize> {
    let value = std::fs::read_to_string(path).ok()?;
    let value = value.trim();
    (value != "max")
        .then(|| value.parse::<usize>().ok())
        .flatten()
        .filter(|value| *value > 0)
}

#[cfg(not(target_os = "linux"))]
pub(crate) const fn effective_memory_limit() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_budget_reserves_process_headroom_and_page_cache() {
        let budget = budget_for_limit(1000, None).unwrap();
        assert_eq!(budget.execution, 640);
        assert_eq!(budget.page_cache, 160);
    }

    #[test]
    fn explicit_page_cache_is_deducted_from_managed_budget() {
        let budget = budget_for_limit(1000, Some(100)).unwrap();
        assert_eq!(budget.execution, 700);
        assert_eq!(budget.page_cache, 100);
        assert_eq!(budget_for_limit(1000, Some(800)), None);
    }
}
