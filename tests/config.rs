//! Configuration: builder values, environment variables, and their precedence.
//!
//! # Why these tests fork a child process
//!
//! `Cargo.toml` sets `[lints.rust] unsafe_code = "forbid"`, and in edition 2024
//! `std::env::set_var`/`remove_var` are `unsafe`. `forbid` cannot be relaxed
//! with `#[allow]`, so a test in this package cannot mutate its own environment.
//!
//! Every env-dependent test therefore re-executes the test binary as a child
//! process with a curated environment (see [`common::run_in_clean_env`]). The
//! child starts with *every* variable the SDK reads removed, so these tests are
//! also immune to whatever the developer happens to have exported — which a
//! `set_var`-based approach would not be.
//!
//! Each such test has two halves: the parent branch spawns the child, and the
//! child branch performs the real assertions.

mod common;

use std::time::Duration;

use rai_sdk::{Config, RetryConfig};

#[cfg(any(feature = "openai", feature = "anthropic", feature = "openrouter"))]
use rai_sdk::Error;

// ── Pure builder behavior (no environment involved) ────────────────────────

#[test]
fn new_config_starts_empty_with_documented_defaults() {
    let config = Config::new();

    assert_eq!(config.openai_api_key, None);
    assert_eq!(config.anthropic_api_key, None);
    assert_eq!(config.openrouter_api_key, None);
    assert_eq!(config.timeout_seconds, None);
    assert_eq!(config.default_max_tokens, None);

    // Documented fallbacks.
    assert_eq!(config.timeout(), 120);
    assert_eq!(config.retry_config().max_retries, 3);
}

#[test]
fn builder_setters_populate_every_field_they_name() {
    let config = Config::new()
        .with_openai_key("openai")
        .with_openai_base_url("https://openai.test")
        .with_anthropic_key("anthropic")
        .with_anthropic_base_url("https://anthropic.test")
        .with_openrouter_key("openrouter")
        .with_openrouter_base_url("https://openrouter.test")
        .with_timeout(7)
        .with_default_max_tokens(512);

    assert_eq!(config.openai_key().as_deref(), Some("openai"));
    assert_eq!(
        config.openai_base_url.as_deref(),
        Some("https://openai.test")
    );
    assert_eq!(config.anthropic_key().as_deref(), Some("anthropic"));
    assert_eq!(
        config.anthropic_base_url.as_deref(),
        Some("https://anthropic.test")
    );
    assert_eq!(config.openrouter_key().as_deref(), Some("openrouter"));
    assert_eq!(
        config.openrouter_base_url().as_deref(),
        Some("https://openrouter.test")
    );
    assert_eq!(config.timeout(), 7);
    assert_eq!(config.default_max_tokens, Some(512));
}

#[test]
fn openrouter_app_url_and_title_aliases_populate_the_canonical_fields() {
    // `with_openrouter_app_url`/`app_title` are legacy aliases; they must keep
    // the canonical referer/title getters working.
    let config = Config::new()
        .with_openrouter_app_url("https://app.example.com")
        .with_openrouter_app_title("Example App");

    assert_eq!(
        config.openrouter_http_referer().as_deref(),
        Some("https://app.example.com")
    );
    assert_eq!(
        config.openrouter_app_url().as_deref(),
        Some("https://app.example.com")
    );
    assert_eq!(config.openrouter_title().as_deref(), Some("Example App"));
    assert_eq!(
        config.openrouter_app_title().as_deref(),
        Some("Example App")
    );
}

#[test]
fn retry_config_getter_returns_defaults_until_one_is_set() {
    assert_eq!(Config::new().retry_config().max_retries, 3);

    let config = Config::new().with_retry_config(
        RetryConfig::new()
            .with_max_retries(9)
            .with_initial_delay(Duration::from_millis(25))
            .with_jitter(false),
    );

    let retry = config.retry_config();
    assert_eq!(retry.max_retries, 9);
    assert_eq!(retry.initial_delay, Duration::from_millis(25));
    assert!(!retry.jitter);
}

#[test]
fn config_serialization_omits_unset_optional_fields() {
    let json = serde_json::to_value(Config::new().with_openai_key("k")).expect("serialize config");
    let object = json.as_object().expect("config serializes to an object");

    assert_eq!(object.keys().collect::<Vec<_>>(), vec!["openai_api_key"]);
}

// ── Missing credentials produce errors, not panics ─────────────────────────

#[cfg(any(feature = "openai", feature = "anthropic", feature = "openrouter"))]
#[test]
fn validation_reports_a_config_error_when_no_key_is_available() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "validation_reports_a_config_error_when_no_key_is_available",
            &[],
        );
        return;
    }

    let config = Config::new();

    #[cfg(feature = "openai")]
    {
        let error = common::expect_error(config.validate_openai());
        assert!(matches!(error, Error::Config(_)), "got {error:?}");
        assert!(
            error.to_string().contains("OPENAI_API_KEY"),
            "the error should name the env var: {error}"
        );
        assert_eq!(error.kind_str(), "config");
    }

    #[cfg(feature = "anthropic")]
    {
        let error = common::expect_error(config.validate_anthropic());
        assert!(matches!(error, Error::Config(_)), "got {error:?}");
        assert!(error.to_string().contains("ANTHROPIC_API_KEY"), "{error}");
    }

    #[cfg(feature = "openrouter")]
    {
        let error = common::expect_error(config.validate_openrouter());
        assert!(matches!(error, Error::Config(_)), "got {error:?}");
        assert!(error.to_string().contains("OPENROUTER_API_KEY"), "{error}");
    }
}

#[test]
fn validation_succeeds_once_a_key_is_supplied_by_the_environment() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "validation_succeeds_once_a_key_is_supplied_by_the_environment",
            &[("OPENAI_API_KEY", "from-env")],
        );
        return;
    }

    // The getters fall back to the environment, so an explicitly empty `Config`
    // still validates when the variable is exported.
    let config = Config::new();
    assert_eq!(config.openai_key().as_deref(), Some("from-env"));

    #[cfg(feature = "openai")]
    assert!(config.validate_openai().is_ok());
}

#[test]
fn a_client_without_credentials_builds_but_exposes_no_providers() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "a_client_without_credentials_builds_but_exposes_no_providers",
            &[],
        );
        return;
    }

    // Client construction must not fail or panic when nothing is configured —
    // the failure surfaces later, per request.
    let client = rai_sdk::Client::new(Config::new()).expect("client should build without keys");

    #[cfg(not(any(feature = "openai", feature = "anthropic", feature = "openrouter")))]
    drop(client);

    #[cfg(feature = "openai")]
    assert!(!client.is_provider_available(rai_sdk::ProviderKind::OpenAI));
    #[cfg(feature = "anthropic")]
    assert!(!client.is_provider_available(rai_sdk::ProviderKind::Anthropic));
    #[cfg(feature = "openrouter")]
    assert!(!client.is_provider_available(rai_sdk::ProviderKind::OpenRouter));
}

#[cfg(feature = "openai")]
#[test]
fn requesting_an_unconfigured_provider_returns_provider_not_configured() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "requesting_an_unconfigured_provider_returns_provider_not_configured",
            &[],
        );
        return;
    }

    let client = rai_sdk::Client::new(Config::new()).expect("client should build without keys");

    let error = tokio::runtime::Runtime::new()
        .expect("build a runtime")
        .block_on(async {
            common::expect_error(
                client
                    .request()
                    .model(rai_sdk::Model::gpt4o_mini())
                    .prompt("hi")
                    .no_retry()
                    .generate()
                    .await,
            )
        });

    assert!(
        matches!(
            error,
            Error::ProviderNotConfigured(rai_sdk::ProviderKind::OpenAI)
        ),
        "got {error:?}"
    );
    assert_eq!(error.provider(), Some(rai_sdk::ProviderKind::OpenAI));
}

// ── `Config::from_env` reads each documented variable ──────────────────────

#[test]
fn from_env_reads_api_keys_and_base_urls() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "from_env_reads_api_keys_and_base_urls",
            &[
                ("OPENAI_API_KEY", "env-openai"),
                ("OPENAI_BASE_URL", "https://env-openai.test"),
                ("ANTHROPIC_API_KEY", "env-anthropic"),
                ("ANTHROPIC_BASE_URL", "https://env-anthropic.test"),
                ("OPENROUTER_API_KEY", "env-openrouter"),
                ("OPENROUTER_BASE_URL", "https://env-openrouter.test"),
            ],
        );
        return;
    }

    let config = Config::from_env();

    assert_eq!(config.openai_api_key.as_deref(), Some("env-openai"));
    assert_eq!(
        config.openai_base_url.as_deref(),
        Some("https://env-openai.test")
    );
    assert_eq!(config.anthropic_api_key.as_deref(), Some("env-anthropic"));
    assert_eq!(
        config.anthropic_base_url.as_deref(),
        Some("https://env-anthropic.test")
    );
    assert_eq!(config.openrouter_api_key.as_deref(), Some("env-openrouter"));
    assert_eq!(
        config.openrouter_base_url().as_deref(),
        Some("https://env-openrouter.test")
    );
}

#[test]
fn from_env_reads_openrouter_attribution_variables() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "from_env_reads_openrouter_attribution_variables",
            &[
                ("OPENROUTER_HTTP_REFERER", "https://env.example.com"),
                ("OPENROUTER_TITLE", "Env Title"),
                ("OPENROUTER_CATEGORIES", " agents, ,productivity "),
            ],
        );
        return;
    }

    let config = Config::from_env();

    assert_eq!(
        config.openrouter_http_referer().as_deref(),
        Some("https://env.example.com")
    );
    assert_eq!(config.openrouter_title().as_deref(), Some("Env Title"));
    // Categories are comma-split, trimmed, and empty entries are dropped.
    assert_eq!(
        config.openrouter_categories(),
        Some(vec!["agents".to_string(), "productivity".to_string()])
    );
}

#[test]
fn from_env_falls_back_to_the_legacy_openrouter_app_variables() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "from_env_falls_back_to_the_legacy_openrouter_app_variables",
            &[
                ("OPENROUTER_APP_URL", "https://legacy.example.com"),
                ("OPENROUTER_APP_TITLE", "Legacy Title"),
            ],
        );
        return;
    }

    let config = Config::from_env();

    assert_eq!(
        config.openrouter_http_referer().as_deref(),
        Some("https://legacy.example.com")
    );
    assert_eq!(config.openrouter_title().as_deref(), Some("Legacy Title"));
}

#[test]
fn the_canonical_openrouter_variables_win_over_the_legacy_ones() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "the_canonical_openrouter_variables_win_over_the_legacy_ones",
            &[
                ("OPENROUTER_HTTP_REFERER", "https://canonical.example.com"),
                ("OPENROUTER_APP_URL", "https://legacy.example.com"),
                ("OPENROUTER_TITLE", "Canonical Title"),
                ("OPENROUTER_APP_TITLE", "Legacy Title"),
            ],
        );
        return;
    }

    let config = Config::from_env();

    assert_eq!(
        config.openrouter_http_referer().as_deref(),
        Some("https://canonical.example.com")
    );
    assert_eq!(
        config.openrouter_title().as_deref(),
        Some("Canonical Title")
    );
}

#[test]
fn from_env_reads_the_request_timeout() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "from_env_reads_the_request_timeout",
            &[("AI_TIMEOUT_SECONDS", "45")],
        );
        return;
    }

    assert_eq!(Config::from_env().timeout(), 45);
}

#[test]
fn from_env_ignores_an_unparseable_timeout_and_keeps_the_default() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "from_env_ignores_an_unparseable_timeout_and_keeps_the_default",
            &[("AI_TIMEOUT_SECONDS", "not-a-number")],
        );
        return;
    }

    // Malformed values are ignored rather than causing a panic or an error.
    let config = Config::from_env();
    assert_eq!(config.timeout_seconds, None);
    assert_eq!(config.timeout(), 120);
}

#[test]
fn from_env_reads_every_retry_variable() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "from_env_reads_every_retry_variable",
            &[
                ("AI_MAX_RETRIES", "6"),
                ("AI_RETRY_INITIAL_DELAY_MS", "250"),
                ("AI_RETRY_MAX_DELAY_MS", "9000"),
                ("AI_RETRY_BACKOFF_MULTIPLIER", "1.5"),
                ("AI_RETRY_JITTER", "off"),
            ],
        );
        return;
    }

    let retry = Config::from_env().retry_config();

    assert_eq!(retry.max_retries, 6);
    assert_eq!(retry.initial_delay, Duration::from_millis(250));
    assert_eq!(retry.max_delay, Duration::from_millis(9000));
    assert!((retry.backoff_multiplier - 1.5).abs() < f64::EPSILON);
    assert!(!retry.jitter);
}

#[test]
fn a_single_retry_variable_customizes_only_that_setting() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "a_single_retry_variable_customizes_only_that_setting",
            &[("AI_MAX_RETRIES", "1")],
        );
        return;
    }

    let retry = Config::from_env().retry_config();

    assert_eq!(retry.max_retries, 1);
    // Untouched settings keep their defaults.
    assert_eq!(retry.initial_delay, Duration::from_secs(1));
    assert_eq!(retry.max_delay, Duration::from_secs(60));
    assert!(retry.jitter);
}

#[test]
fn retry_jitter_accepts_every_documented_boolean_spelling() {
    if !common::in_env_child() {
        for value in ["1", "true", "yes", "on", "TRUE", " On "] {
            common::run_in_clean_env(
                "retry_jitter_accepts_every_documented_boolean_spelling",
                &[("AI_RETRY_JITTER", value), ("RAI_EXPECT_JITTER", "1")],
            );
        }
        for value in ["0", "false", "no", "off", "FALSE", " Off "] {
            common::run_in_clean_env(
                "retry_jitter_accepts_every_documented_boolean_spelling",
                &[("AI_RETRY_JITTER", value), ("RAI_EXPECT_JITTER", "0")],
            );
        }
        return;
    }

    let expected = std::env::var("RAI_EXPECT_JITTER").expect("parent sets the expectation") == "1";
    let raw = std::env::var("AI_RETRY_JITTER").expect("parent sets the value under test");

    assert_eq!(
        Config::from_env().retry_config().jitter,
        expected,
        "AI_RETRY_JITTER={raw:?} should parse to {expected}"
    );
}

#[test]
fn an_unrecognized_jitter_value_leaves_the_default_in_place() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "an_unrecognized_jitter_value_leaves_the_default_in_place",
            &[("AI_RETRY_JITTER", "maybe")],
        );
        return;
    }

    // Unparseable booleans are ignored, and because no other retry variable was
    // set the whole retry config stays at its default.
    let config = Config::from_env();
    assert!(config.retry_config.is_none());
    assert!(config.retry_config().jitter);
}

#[test]
fn no_retry_variables_means_no_stored_retry_config() {
    if !common::in_env_child() {
        common::run_in_clean_env("no_retry_variables_means_no_stored_retry_config", &[]);
        return;
    }

    let config = Config::from_env();
    assert!(
        config.retry_config.is_none(),
        "an untouched retry config should not be materialized"
    );
    assert_eq!(config.retry_config().max_retries, 3);
}

// ── Precedence: explicit values beat the environment ──────────────────────

#[test]
fn explicit_builder_values_take_precedence_over_the_environment() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "explicit_builder_values_take_precedence_over_the_environment",
            &[
                ("OPENAI_API_KEY", "env-openai"),
                ("ANTHROPIC_API_KEY", "env-anthropic"),
                ("OPENROUTER_API_KEY", "env-openrouter"),
                ("OPENROUTER_BASE_URL", "https://env.example.com"),
                ("OPENROUTER_HTTP_REFERER", "https://env.example.com"),
                ("OPENROUTER_TITLE", "Env Title"),
                ("OPENROUTER_CATEGORIES", "env-category"),
            ],
        );
        return;
    }

    let config = Config::from_env()
        .with_openai_key("explicit-openai")
        .with_anthropic_key("explicit-anthropic")
        .with_openrouter_key("explicit-openrouter")
        .with_openrouter_base_url("https://explicit.example.com")
        .with_openrouter_http_referer("https://explicit.example.com")
        .with_openrouter_title("Explicit Title")
        .with_openrouter_categories(vec!["explicit-category".to_string()]);

    assert_eq!(config.openai_key().as_deref(), Some("explicit-openai"));
    assert_eq!(
        config.anthropic_key().as_deref(),
        Some("explicit-anthropic")
    );
    assert_eq!(
        config.openrouter_key().as_deref(),
        Some("explicit-openrouter")
    );
    assert_eq!(
        config.openrouter_base_url().as_deref(),
        Some("https://explicit.example.com")
    );
    assert_eq!(
        config.openrouter_http_referer().as_deref(),
        Some("https://explicit.example.com")
    );
    assert_eq!(config.openrouter_title().as_deref(), Some("Explicit Title"));
    assert_eq!(
        config.openrouter_categories(),
        Some(vec!["explicit-category".to_string()])
    );
}

#[test]
fn client_builder_from_env_is_overridden_by_later_explicit_setters() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "client_builder_from_env_is_overridden_by_later_explicit_setters",
            &[
                ("OPENAI_API_KEY", "env-openai"),
                ("AI_MAX_RETRIES", "5"),
                ("AI_TIMEOUT_SECONDS", "10"),
            ],
        );
        return;
    }

    let client = rai_sdk::Client::builder()
        .from_env()
        .openai_key("explicit-openai")
        .timeout(30)
        .build()
        .expect("client should build");

    assert_eq!(
        client.config().openai_key().as_deref(),
        Some("explicit-openai")
    );
    assert_eq!(client.config().timeout(), 30);
    // `from_env` seeded the retry config, which the explicit setters left alone.
    assert_eq!(client.config().retry_config().max_retries, 5);
}

#[test]
fn client_builder_from_env_does_not_inherit_variables_set_after_it_runs() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "client_builder_from_env_does_not_inherit_variables_set_after_it_runs",
            &[("OPENAI_API_KEY", "env-openai")],
        );
        return;
    }

    // `from_env` snapshots the environment into the config eagerly; the stored
    // field, not a later lookup, is what the client uses.
    let client = rai_sdk::Client::builder()
        .from_env()
        .build()
        .expect("client should build");

    assert_eq!(
        client.config().openai_api_key.as_deref(),
        Some("env-openai"),
        "from_env should copy the value into the config field"
    );
}

#[test]
fn client_builder_no_retry_overrides_retry_variables_from_the_environment() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "client_builder_no_retry_overrides_retry_variables_from_the_environment",
            &[("AI_MAX_RETRIES", "7")],
        );
        return;
    }

    // `from_env` seeds the builder's default retry config from AI_MAX_RETRIES,
    // and `no_retry()` afterwards must win.
    let config_retries = Config::from_env().retry_config().max_retries;
    assert_eq!(config_retries, 7, "sanity check on the env-derived value");

    let client = rai_sdk::Client::builder()
        .from_env()
        .openai_key("explicit")
        .no_retry()
        .build()
        .expect("client should build");

    // The config still records the env-derived value...
    assert_eq!(client.config().retry_config().max_retries, 7);
    // ...but requests use the builder default, which `no_retry()` zeroed. That
    // is asserted end-to-end against a mock server in `tests/retry.rs`.
}

// ── The OpenAI-compatible endpoint is per client, never from the env ───────

#[test]
fn openai_compatible_settings_are_never_read_from_the_environment() {
    if !common::in_env_child() {
        common::run_in_clean_env(
            "openai_compatible_settings_are_never_read_from_the_environment",
            &[
                ("OPENAI_API_KEY", "sk-from-env"),
                ("OPENAI_BASE_URL", "https://proxy.example.com/v1"),
            ],
        );
        return;
    }

    let config = Config::from_env();

    // `OPENAI_BASE_URL` keeps its existing meaning: it redirects the real
    // OpenAI provider and nothing else.
    assert_eq!(
        config.openai_base_url.as_deref(),
        Some("https://proxy.example.com/v1")
    );

    // A process-wide variable cannot conjure a compatible endpoint, because a
    // process may need several with different credentials and capabilities.
    assert_eq!(config.openai_compatible_base_url(), None);
    assert_eq!(config.openai_compatible_key(), None);

    // Setting one explicitly leaves the OpenAI provider's own base URL alone.
    let config = config.with_ollama();
    assert_eq!(
        config.openai_compatible_base_url().as_deref(),
        Some(rai_sdk::config::OLLAMA_BASE_URL)
    );
    assert_eq!(
        config.openai_base_url.as_deref(),
        Some("https://proxy.example.com/v1")
    );
}

#[test]
fn openai_compatible_capabilities_default_to_full_compatibility() {
    let config = Config::new();
    assert_eq!(
        config.openai_compatible_capabilities(),
        rai_sdk::EndpointCapabilities::all()
    );

    let config = config.with_openai_compatible_capabilities(
        rai_sdk::EndpointCapabilities::text_only().with_structured_output(true),
    );
    let capabilities = config.openai_compatible_capabilities();
    assert!(!capabilities.tool_calling);
    assert!(capabilities.structured_output);
}
