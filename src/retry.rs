//! Retry policy for transient provider failures.
//!
//! [`RetryConfig`] describes exponential backoff with optional jitter. It is
//! applied automatically around provider calls, and only to errors that
//! [`Error::is_retryable`](crate::Error::is_retryable) reports as transient
//! (rate limits, timeouts, and transport-level HTTP errors). Non-transient
//! errors are returned immediately, so a bad request never sleeps.
//!
//! The delay for attempt `n` is `initial_delay * backoff_multiplier^n`, clamped
//! to `max_delay`; with jitter enabled, a random offset of up to 50% of that
//! value is added.

use std::time::Duration;

use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Configuration for automatic retry with exponential backoff.
///
/// Applied to retryable errors (`RateLimit`, `Timeout`, `Http`).
///
/// Attach it to a client with
/// [`ClientBuilder::retry_config`](crate::ClientBuilder::retry_config) or to a
/// single request with
/// [`RequestBuilder::retry_config`](crate::RequestBuilder::retry_config).
///
/// # Examples
///
/// ```rust
/// use std::time::Duration;
/// use rai_sdk::RetryConfig;
///
/// // Default: 3 retries, 1s initial delay, 2x backoff, jitter enabled
/// let config = RetryConfig::default();
///
/// // Custom: 5 retries, 500ms initial delay, no jitter
/// let custom = RetryConfig::new()
///     .with_max_retries(5)
///     .with_initial_delay(Duration::from_millis(500))
///     .with_jitter(false);
///
/// // Disable retries entirely
/// let none = RetryConfig::none();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Delay before the first retry.
    pub initial_delay: Duration,
    /// Maximum delay between retries (caps exponential growth).
    pub max_delay: Duration,
    /// Multiplier applied to the delay after each attempt.
    pub backoff_multiplier: f64,
    /// Add random jitter to avoid thundering herd.
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Create a default retry configuration (3 retries, 1s initial delay, 2x backoff).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration that disables retries entirely.
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    /// Set the maximum number of retry attempts.
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Set the initial delay before the first retry.
    pub fn with_initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set the maximum delay between retries.
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set the backoff multiplier.
    pub fn with_backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Enable or disable jitter.
    pub fn with_jitter(mut self, jitter: bool) -> Self {
        self.jitter = jitter;
        self
    }

    /// Compute the delay for a given attempt (0-indexed).
    pub(crate) fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base = self.initial_delay.as_secs_f64() * self.backoff_multiplier.powi(attempt as i32);
        let capped = base.min(self.max_delay.as_secs_f64());

        let final_delay = if self.jitter {
            let jitter_range = capped * 0.5;
            let jitter_offset = rand::rng().random_range(0.0..jitter_range);
            capped + jitter_offset
        } else {
            capped
        };

        Duration::from_secs_f64(final_delay)
    }
}

/// Execute an async operation with retry logic.
///
/// The `operation` closure is called repeatedly until it succeeds or the retry
/// limit is exceeded. Only errors where `Error::is_retryable()` returns `true`
/// are retried.
pub(crate) async fn with_retry<F, Fut, T>(
    config: &RetryConfig,
    operation_name: &str,
    mut operation: F,
) -> crate::error::Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = crate::error::Result<T>>,
{
    for attempt in 0..=config.max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if !e.is_retryable() || attempt == config.max_retries {
                    return Err(e);
                }

                let delay = config.delay_for_attempt(attempt);
                warn!(
                    operation = operation_name,
                    attempt = attempt + 1,
                    max_retries = config.max_retries,
                    delay_ms = delay.as_millis() as u64,
                    error_kind = e.kind_str(),
                    provider = ?e.provider(),
                    error = %e,
                    "Retrying after transient error"
                );

                tokio::time::sleep(delay).await;
            }
        }
    }

    // Unreachable: the loop always returns on the final attempt.
    unreachable!("retry loop should have returned")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_delay, Duration::from_secs(1));
        assert_eq!(config.max_delay, Duration::from_secs(60));
        assert!((config.backoff_multiplier - 2.0).abs() < f64::EPSILON);
        assert!(config.jitter);
    }

    #[test]
    fn none_config() {
        let config = RetryConfig::none();
        assert_eq!(config.max_retries, 0);
    }

    #[test]
    fn delay_exponential_growth() {
        let config = RetryConfig::new().with_jitter(false);

        let d0 = config.delay_for_attempt(0);
        let d1 = config.delay_for_attempt(1);
        let d2 = config.delay_for_attempt(2);

        assert_eq!(d0, Duration::from_secs(1)); // 1 * 2^0 = 1s
        assert_eq!(d1, Duration::from_secs(2)); // 1 * 2^1 = 2s
        assert_eq!(d2, Duration::from_secs(4)); // 1 * 2^2 = 4s
    }

    #[test]
    fn delay_capped_at_max() {
        let config = RetryConfig::new()
            .with_jitter(false)
            .with_max_delay(Duration::from_secs(3));

        let d0 = config.delay_for_attempt(0); // 1s
        let d1 = config.delay_for_attempt(1); // 2s
        let d2 = config.delay_for_attempt(2); // 4s -> capped to 3s
        let d3 = config.delay_for_attempt(3); // 8s -> capped to 3s

        assert_eq!(d0, Duration::from_secs(1));
        assert_eq!(d1, Duration::from_secs(2));
        assert_eq!(d2, Duration::from_secs(3));
        assert_eq!(d3, Duration::from_secs(3));
    }

    #[test]
    fn delay_with_jitter_is_bounded() {
        let config = RetryConfig::new().with_jitter(true);

        // With jitter, delay should be in [base, base * 1.5)
        for attempt in 0..5 {
            let base =
                config.initial_delay.as_secs_f64() * config.backoff_multiplier.powi(attempt as i32);
            let capped = base.min(config.max_delay.as_secs_f64());

            let delay = config.delay_for_attempt(attempt);
            let delay_secs = delay.as_secs_f64();

            assert!(
                delay_secs >= capped,
                "attempt {attempt}: delay {delay_secs} < base {capped}"
            );
            assert!(
                delay_secs < capped * 1.5,
                "attempt {attempt}: delay {delay_secs} >= max {:.2}",
                capped * 1.5
            );
        }
    }

    #[test]
    fn builder_chain() {
        let config = RetryConfig::new()
            .with_max_retries(5)
            .with_initial_delay(Duration::from_millis(500))
            .with_max_delay(Duration::from_secs(30))
            .with_backoff_multiplier(3.0)
            .with_jitter(false);

        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_delay, Duration::from_millis(500));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert!((config.backoff_multiplier - 3.0).abs() < f64::EPSILON);
        assert!(!config.jitter);
    }
}
