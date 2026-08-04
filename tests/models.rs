//! Model selection: provider association, exact wire model IDs, and serde shape.
//!
//! These assertions pin the strings that actually go over the wire. Changing a
//! model ID is a breaking change for callers, so it should require editing a
//! test on purpose.

use rai_sdk::{AnthropicModel, Model, OpenAIModel, OpenRouterModel, ProviderKind};

/// Assert `constructor().as_str()` and the provider it routes to.
fn assert_model(model: Model, expected_id: &str, expected_provider: ProviderKind) {
    assert_eq!(
        model.as_str(),
        expected_id,
        "wire model ID changed for {model:?}"
    );
    assert_eq!(
        model.provider(),
        expected_provider,
        "provider association changed for {model:?}"
    );
}

#[test]
fn openai_constructors_map_to_expected_wire_ids() {
    let cases = [
        (Model::gpt4o(), "gpt-4o"),
        (Model::gpt4o_mini(), "gpt-4o-mini"),
        (Model::gpt4_1(), "gpt-4.1"),
        (Model::o3_mini(), "o3-mini"),
        (Model::o3(), "o3"),
        (Model::o4_mini(), "o4-mini"),
        (Model::gpt5(), "gpt-5"),
        (Model::gpt5_mini(), "gpt-5-mini"),
        (Model::gpt5_nano(), "gpt-5-nano"),
        (Model::gpt5_codex(), "gpt-5-codex"),
        (Model::gpt_5_1(), "gpt-5.1"),
        (Model::gpt_5_2(), "gpt-5.2"),
        (Model::gpt_5_2_pro(), "gpt-5.2-pro"),
        (Model::gpt_5_3_chat(), "gpt-5.3-chat"),
        (Model::gpt_5_3_instant(), "gpt-5.3-instant"),
        (Model::gpt_5_4(), "gpt-5.4"),
        (Model::gpt_5_4_mini(), "gpt-5.4-mini"),
        (Model::gpt_5_4_nano(), "gpt-5.4-nano"),
        (Model::gpt_5_5(), "gpt-5.5"),
    ];

    for (model, expected_id) in cases {
        assert_model(model, expected_id, ProviderKind::OpenAI);
    }
}

#[test]
fn anthropic_constructors_map_to_expected_wire_ids() {
    let cases = [
        (Model::claude_fable_5(), "claude-fable-5"),
        (Model::claude_opus_48(), "claude-opus-4-8"),
        (Model::claude_opus_47(), "claude-opus-4-7"),
        (Model::claude_opus_46(), "claude-opus-4-6"),
        (Model::claude_sonnet_46(), "claude-sonnet-4-6"),
        (Model::claude_sonnet_45(), "claude-sonnet-4-5"),
        (Model::claude_opus_45(), "claude-opus-4-5"),
        (Model::claude_haiku_45(), "claude-haiku-4-5"),
        (Model::claude_opus_41(), "claude-opus-4-1"),
        (Model::claude_sonnet_4(), "claude-sonnet-4-0"),
        (Model::claude_opus_4(), "claude-opus-4-0"),
        (Model::claude_35_sonnet(), "claude-3-5-sonnet-20241022"),
        (Model::claude_35_haiku(), "claude-3-5-haiku-20241022"),
    ];

    for (model, expected_id) in cases {
        assert_model(model, expected_id, ProviderKind::Anthropic);
    }
}

#[test]
fn anthropic_four_series_ids_use_dashes_not_dots() {
    // Anthropic's own API uses `claude-sonnet-4-6`, while the same model on
    // OpenRouter is `anthropic/claude-sonnet-4.6`. Mixing them up produces a
    // 404 from the provider, so the distinction is worth pinning.
    assert_eq!(Model::claude_sonnet_46().as_str(), "claude-sonnet-4-6");
    assert_eq!(
        Model::openrouter_custom("anthropic/claude-sonnet-4.6").as_str(),
        "anthropic/claude-sonnet-4.6"
    );
}

#[test]
fn curated_openrouter_constructors_map_to_expected_wire_ids() {
    let cases = [
        (Model::openrouter_auto(), "openrouter/auto"),
        (Model::openrouter_gpt5(), "openai/gpt-5"),
        (
            Model::openrouter_claude_sonnet_4_5(),
            "anthropic/claude-sonnet-4.5",
        ),
        (
            Model::openrouter_gemini_25_flash(),
            "google/gemini-2.5-flash",
        ),
        (Model::openrouter_deepseek_r1(), "deepseek/deepseek-r1"),
        (Model::openrouter_qwen3_coder(), "qwen/qwen3-coder"),
    ];

    for (model, expected_id) in cases {
        assert_model(model, expected_id, ProviderKind::OpenRouter);
    }
}

#[test]
fn openrouter_ids_are_vendor_prefixed() {
    // Every curated OpenRouter ID must be a `vendor/model` path; a bare model
    // name would be rejected by OpenRouter.
    let ids = [
        OpenRouterModel::Auto.as_str(),
        OpenRouterModel::Free.as_str(),
        OpenRouterModel::Gpt5_5.as_str(),
        OpenRouterModel::ClaudeSonnet4_6.as_str(),
        OpenRouterModel::Gemini35Flash.as_str(),
        OpenRouterModel::Grok4_3.as_str(),
        OpenRouterModel::Llama4Scout.as_str(),
        OpenRouterModel::Qwen3Max.as_str(),
        OpenRouterModel::DeepseekV4Pro.as_str(),
        OpenRouterModel::MistralLarge2512.as_str(),
        OpenRouterModel::SonarPro.as_str(),
        OpenRouterModel::CommandA.as_str(),
        OpenRouterModel::KimiK2_7Code.as_str(),
        OpenRouterModel::Glm5.as_str(),
        OpenRouterModel::MimoV2_5Pro.as_str(),
    ];

    for id in ids {
        assert!(
            id.split('/').count() == 2,
            "OpenRouter model ID `{id}` should be a single `vendor/model` path"
        );
        assert!(
            !id.starts_with('/') && !id.ends_with('/'),
            "OpenRouter model ID `{id}` has an empty vendor or model segment"
        );
    }
}

#[test]
fn custom_model_constructors_pass_the_id_through_verbatim() {
    assert_model(
        Model::openai_custom("ft:gpt-4o-mini:acme::abc123"),
        "ft:gpt-4o-mini:acme::abc123",
        ProviderKind::OpenAI,
    );
    assert_model(
        Model::anthropic_custom("claude-experimental-9"),
        "claude-experimental-9",
        ProviderKind::Anthropic,
    );
    assert_model(
        Model::openrouter_custom("acme/private-model:free"),
        "acme/private-model:free",
        ProviderKind::OpenRouter,
    );
}

#[test]
fn openrouter_from_str_falls_back_to_custom_for_unknown_ids() {
    assert_eq!(
        "openrouter/auto".parse::<OpenRouterModel>(),
        Ok(OpenRouterModel::Auto)
    );
    assert_eq!(
        "anthropic/claude-sonnet-4.6".parse::<OpenRouterModel>(),
        Ok(OpenRouterModel::ClaudeSonnet4_6)
    );
    assert_eq!(
        "vendor-that-does-not-exist/model".parse::<OpenRouterModel>(),
        Ok(OpenRouterModel::Custom(
            "vendor-that-does-not-exist/model".to_string()
        ))
    );

    // Parsing is infallible, so even nonsense round-trips as a custom ID.
    assert_eq!(
        "".parse::<OpenRouterModel>(),
        Ok(OpenRouterModel::Custom(String::new()))
    );
}

#[test]
fn every_curated_openrouter_id_round_trips_through_from_str() {
    // Guards against a typo that makes `as_str` and `from_str` disagree, which
    // would silently demote a curated variant to `Custom`.
    let variants = [
        OpenRouterModel::Auto,
        OpenRouterModel::Free,
        OpenRouterModel::Gpt5,
        OpenRouterModel::Gpt5Mini,
        OpenRouterModel::Gpt5Nano,
        OpenRouterModel::Gpt5Codex,
        OpenRouterModel::Gpt5_1,
        OpenRouterModel::Gpt5_2,
        OpenRouterModel::Gpt5_2Pro,
        OpenRouterModel::Gpt5_3Chat,
        OpenRouterModel::Gpt5_4,
        OpenRouterModel::Gpt5_4Mini,
        OpenRouterModel::Gpt5_4Nano,
        OpenRouterModel::Gpt5_4Pro,
        OpenRouterModel::Gpt5_5,
        OpenRouterModel::Gpt5_5Pro,
        OpenRouterModel::Gpt4_1,
        OpenRouterModel::Gpt4o,
        OpenRouterModel::O3,
        OpenRouterModel::O3Pro,
        OpenRouterModel::O3DeepResearch,
        OpenRouterModel::O4Mini,
        OpenRouterModel::GptOss120b,
        OpenRouterModel::ClaudeFable5,
        OpenRouterModel::ClaudeSonnet4,
        OpenRouterModel::ClaudeSonnet4_5,
        OpenRouterModel::ClaudeSonnet4_6,
        OpenRouterModel::ClaudeOpus4_1,
        OpenRouterModel::ClaudeOpus4_5,
        OpenRouterModel::ClaudeOpus4_6,
        OpenRouterModel::ClaudeOpus4_6Fast,
        OpenRouterModel::ClaudeOpus4_7,
        OpenRouterModel::ClaudeOpus4_7Fast,
        OpenRouterModel::ClaudeOpus4_8,
        OpenRouterModel::ClaudeOpus4_8Fast,
        OpenRouterModel::ClaudeHaiku4_5,
        OpenRouterModel::Claude3_7Sonnet,
        OpenRouterModel::Gemini35Flash,
        OpenRouterModel::Gemini31ProPreview,
        OpenRouterModel::Gemini31ProPreviewCustomTools,
        OpenRouterModel::Gemini31FlashLite,
        OpenRouterModel::Gemini31FlashLitePreview,
        OpenRouterModel::Gemini31FlashImagePreview,
        OpenRouterModel::Gemini3ProImagePreview,
        OpenRouterModel::Gemini3FlashPreview,
        OpenRouterModel::Gemini25Pro,
        OpenRouterModel::Gemini25Flash,
        OpenRouterModel::Gemini25FlashImage,
        OpenRouterModel::Grok4_3,
        OpenRouterModel::Grok4_20,
        OpenRouterModel::Grok4_20MultiAgent,
        OpenRouterModel::GrokBuild0_1,
        OpenRouterModel::Grok4,
        OpenRouterModel::Grok4Fast,
        OpenRouterModel::Grok4_1Fast,
        OpenRouterModel::GrokCodeFast1,
        OpenRouterModel::Llama4Maverick,
        OpenRouterModel::Llama4Scout,
        OpenRouterModel::Llama3_3_70bInstruct,
        OpenRouterModel::Llama3_2_11bVisionInstruct,
        OpenRouterModel::Qwen3Max,
        OpenRouterModel::Qwen3MaxThinking,
        OpenRouterModel::Qwen3Coder,
        OpenRouterModel::Qwen3CoderPlus,
        OpenRouterModel::Qwen3_235bA22b,
        OpenRouterModel::Qwen3Vl235bA22bInstruct,
        OpenRouterModel::Qwen3Vl235bA22bThinking,
        OpenRouterModel::Qwen3_7Max,
        OpenRouterModel::Qwen3_7Plus,
        OpenRouterModel::Qwen3_6MaxPreview,
        OpenRouterModel::Qwen3_6Plus,
        OpenRouterModel::Qwen3_6Flash,
        OpenRouterModel::DeepseekChatV3_1,
        OpenRouterModel::DeepseekR1,
        OpenRouterModel::DeepseekV3_2,
        OpenRouterModel::DeepseekV4Flash,
        OpenRouterModel::DeepseekV4Pro,
        OpenRouterModel::MistralLarge,
        OpenRouterModel::MistralMedium3_1,
        OpenRouterModel::Codestral2508,
        OpenRouterModel::DevstralMedium,
        OpenRouterModel::PixtralLarge2411,
        OpenRouterModel::MistralLarge2512,
        OpenRouterModel::MistralMedium3_5,
        OpenRouterModel::Devstral2512,
        OpenRouterModel::Ministral14b2512,
        OpenRouterModel::SonarPro,
        OpenRouterModel::SonarReasoningPro,
        OpenRouterModel::SonarDeepResearch,
        OpenRouterModel::CommandA,
        OpenRouterModel::KimiK2_5,
        OpenRouterModel::KimiK2Thinking,
        OpenRouterModel::KimiK2_6,
        OpenRouterModel::KimiK2_7Code,
        OpenRouterModel::Glm5,
        OpenRouterModel::Glm5_1,
        OpenRouterModel::Glm5Turbo,
        OpenRouterModel::Glm4_7,
        OpenRouterModel::MimoV2Omni,
        OpenRouterModel::MimoV2Flash,
        OpenRouterModel::MimoV2_5,
        OpenRouterModel::MimoV2_5Pro,
    ];

    for variant in variants {
        let id = variant.as_str().to_string();
        assert_eq!(
            id.parse::<OpenRouterModel>(),
            Ok(variant.clone()),
            "`{id}` did not round-trip back to {variant:?}"
        );
    }
}

#[test]
fn only_o_series_models_are_treated_as_reasoning_models() {
    let reasoning = [
        OpenAIModel::O1Preview,
        OpenAIModel::O1Mini,
        OpenAIModel::O3Mini,
        OpenAIModel::O3,
        OpenAIModel::O4Mini,
    ];
    for model in reasoning {
        assert!(
            model.is_reasoning_model(),
            "{model:?} should be a reasoning model"
        );
    }

    let non_reasoning = [
        OpenAIModel::Gpt4o,
        OpenAIModel::Gpt4oMini,
        OpenAIModel::Gpt4Turbo,
        OpenAIModel::Gpt4_1,
        OpenAIModel::Gpt5,
        OpenAIModel::Gpt5_5,
        OpenAIModel::Custom("o3-lookalike".to_string()),
    ];
    for model in non_reasoning {
        assert!(
            !model.is_reasoning_model(),
            "{model:?} should not be a reasoning model"
        );
    }
}

#[test]
fn model_serde_uses_an_adjacently_tagged_provider_representation() {
    let json = serde_json::to_value(Model::gpt4o_mini()).expect("serialize model");
    assert_eq!(
        json,
        serde_json::json!({ "provider": "OpenAI", "model": "Gpt4oMini" })
    );

    let custom = serde_json::to_value(Model::openrouter_custom("acme/model")).expect("serialize");
    assert_eq!(
        custom,
        serde_json::json!({
            "provider": "OpenRouter",
            "model": { "Custom": "acme/model" }
        })
    );

    let round_tripped: Model = serde_json::from_value(custom).expect("deserialize model");
    assert_eq!(round_tripped.as_str(), "acme/model");
    assert_eq!(round_tripped.provider(), ProviderKind::OpenRouter);
}

#[test]
fn provider_kind_display_and_serde_are_lowercase() {
    assert_eq!(ProviderKind::OpenAI.to_string(), "openai");
    assert_eq!(ProviderKind::Anthropic.to_string(), "anthropic");
    assert_eq!(ProviderKind::OpenRouter.to_string(), "openrouter");

    assert_eq!(
        serde_json::to_value(ProviderKind::OpenRouter).expect("serialize provider"),
        serde_json::json!("openrouter")
    );
    assert_eq!(
        serde_json::from_value::<ProviderKind>(serde_json::json!("anthropic"))
            .expect("deserialize provider"),
        ProviderKind::Anthropic
    );
}

#[test]
fn anthropic_and_openai_model_ids_are_not_vendor_prefixed() {
    // Direct provider APIs take bare model names. A stray `vendor/` prefix here
    // would mean an OpenRouter ID leaked into a first-party model list.
    let direct_ids = [
        AnthropicModel::ClaudeSonnet46.as_str(),
        AnthropicModel::Claude3Opus.as_str(),
        OpenAIModel::Gpt5_5.as_str(),
        OpenAIModel::Gpt4o.as_str(),
    ];

    for id in direct_ids {
        assert!(
            !id.contains('/'),
            "first-party model ID `{id}` should not be vendor-prefixed"
        );
    }
}
