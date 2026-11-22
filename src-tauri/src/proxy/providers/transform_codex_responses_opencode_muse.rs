//! OpenCode Muse native Responses compatibility.
//!
//! OpenCode has absorbed several Codex extensions that used to require a local
//! rewrite: top-level `namespace` tools, `tool_search_output`, optional
//! `tool_search` parameters, and `tool_search` alongside structured output.
//! Preserve those shapes here. This module only normalizes request features
//! that the OpenCode Muse validator still rejects.

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};

const UNSUPPORTED_TOP_LEVEL_FIELDS: &[&str] = &["access_programs", "external_web_access"];
const UNSUPPORTED_INCLUDE_MARKERS: &[&str] =
    &["message.output_text.logprobs", "output_text.logprobs"];
const ORPHAN_TOOL_OUTPUT_MARKER: &str = "[tool_result without a matching tool_call]";
const MESSAGE_CONTENT_PART_TYPES: &[&str] = &["input_text", "input_image", "input_file"];

/// Apply request rewrites required by OpenCode's Muse Responses validator.
///
/// The caller gates this by native Responses transport, `opencode.ai`, and the
/// final outbound model. Requests without one of the affected shapes remain
/// byte-identical.
pub(crate) fn apply_opencode_muse_request_compat(body: &mut Value) -> bool {
    if !body.is_object() {
        return false;
    }

    let mut changed = normalize_call_id_less_tool_outputs(body);
    changed |= normalize_reasoning_summary(body);

    if let Some(root) = body.as_object_mut() {
        for field in UNSUPPORTED_TOP_LEVEL_FIELDS {
            changed |= root.remove(*field).is_some();
        }
    }
    changed |= strip_unsupported_include_markers(body);
    changed |= normalize_replayed_function_call_arguments(body);

    if is_compaction_summary_turn(body) {
        // The existing native-compaction bridge has already turned the private
        // trigger into a text-only prompt and set `tool_choice: "none"`, which
        // Muse rejects. Muse has accepted replaying tool history without tool
        // declarations, so remove the declaration and both choice controls.
        changed |= remove_additional_tool_carriers(body, true);
        if let Some(root) = body.as_object_mut() {
            changed |= root.remove("tools").is_some();
            changed |= root.remove("tool_choice").is_some();
            changed |= root.remove("parallel_tool_calls").is_some();
        }
        return changed;
    }

    changed |= promote_additional_tools(body);
    changed |= normalize_function_tool_schemas(body);
    changed |= normalize_tool_search_schemas(body);
    changed |= normalize_tool_choice(body);
    changed
}

fn json_type(value: &Value) -> Option<&str> {
    value.get("type").and_then(Value::as_str).map(str::trim)
}

fn valid_call_id(value: &Value) -> Option<&str> {
    value
        .get("call_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|call_id| !call_id.is_empty())
}

fn output_type_for_call(call_type: &str) -> Option<&'static str> {
    match call_type {
        "function_call" => Some("function_call_output"),
        "custom_tool_call" => Some("custom_tool_call_output"),
        _ => None,
    }
}

fn output_key(output_type: &str, call_id: &str) -> String {
    format!("{output_type}\0{call_id}")
}

/// Attach call-id-less receipts to their only pending call of the same kind.
/// Receipts that still have no unique owner are preserved as user context.
fn normalize_call_id_less_tool_outputs(body: &mut Value) -> bool {
    let Some(input) = body.get_mut("input").and_then(Value::as_array_mut) else {
        return false;
    };

    let mut responded = HashSet::new();
    for item in input.iter() {
        if let Some(output_type) = json_type(item) {
            if matches!(
                output_type,
                "function_call_output" | "custom_tool_call_output"
            ) {
                if let Some(call_id) = valid_call_id(item) {
                    responded.insert(output_key(output_type, call_id));
                }
            }
        }
    }

    let mut pending = HashMap::from([
        ("function_call_output", Vec::new()),
        ("custom_tool_call_output", Vec::new()),
    ]);
    for item in input.iter() {
        let Some(call_type) = json_type(item) else {
            continue;
        };
        let Some(output_type) = output_type_for_call(call_type) else {
            continue;
        };
        let Some(call_id) = valid_call_id(item) else {
            continue;
        };
        if !responded.contains(&output_key(output_type, call_id)) {
            pending
                .get_mut(output_type)
                .expect("known output type")
                .push(call_id.to_string());
        }
    }

    let mut changed = false;
    for item in input.iter_mut() {
        let Some(output_type) = json_type(item) else {
            continue;
        };
        if !matches!(
            output_type,
            "function_call_output" | "custom_tool_call_output"
        ) {
            continue;
        }
        if valid_call_id(item).is_some() {
            continue;
        }

        let call_id = pending
            .get_mut(output_type)
            .and_then(|candidates| (candidates.len() == 1).then(|| candidates.remove(0)));
        if let Some(call_id) = call_id {
            if let Some(object) = item.as_object_mut() {
                object.insert("call_id".to_string(), Value::String(call_id));
                changed = true;
                continue;
            }
        }

        *item = json!({
            "type": "message",
            "role": "user",
            "content": orphan_tool_output_content(item.get("output")),
        });
        changed = true;
    }
    changed
}

fn orphan_tool_output_content(output: Option<&Value>) -> Vec<Value> {
    match output {
        Some(Value::String(text)) if !text.is_empty() => {
            vec![input_text_part(&format!(
                "{ORPHAN_TOOL_OUTPUT_MARKER}\n{text}"
            ))]
        }
        Some(Value::Array(parts)) => {
            let mut content = vec![input_text_part(ORPHAN_TOOL_OUTPUT_MARKER)];
            content.extend(parts.iter().map(|part| {
                if json_type(part).is_some_and(|ty| MESSAGE_CONTENT_PART_TYPES.contains(&ty)) {
                    part.clone()
                } else {
                    input_text_part(&part.to_string())
                }
            }));
            content
        }
        Some(Value::Null | Value::String(_)) | None => {
            vec![input_text_part(ORPHAN_TOOL_OUTPUT_MARKER)]
        }
        Some(other) => vec![input_text_part(&format!(
            "{ORPHAN_TOOL_OUTPUT_MARKER}\n{other}"
        ))],
    }
}

fn input_text_part(text: &str) -> Value {
    json!({ "type": "input_text", "text": text })
}

fn normalize_reasoning_summary(body: &mut Value) -> bool {
    let Some(reasoning) = body.get_mut("reasoning").and_then(Value::as_object_mut) else {
        return false;
    };
    let Some(summary) = reasoning.get("summary") else {
        return false;
    };
    if summary
        .as_str()
        .is_some_and(|value| matches!(value, "auto" | "concise" | "detailed"))
    {
        return false;
    }
    reasoning.remove("summary");
    true
}

fn strip_unsupported_include_markers(body: &mut Value) -> bool {
    let mut removed_marker = false;
    let mut remove_include = false;
    if let Some(include) = body.get_mut("include").and_then(Value::as_array_mut) {
        let before = include.len();
        include.retain(|marker| {
            !marker
                .as_str()
                .is_some_and(|marker| UNSUPPORTED_INCLUDE_MARKERS.contains(&marker))
        });
        removed_marker = include.len() != before;
        remove_include = removed_marker && include.is_empty();
    }
    if let Some(root) = body.as_object_mut() {
        if remove_include {
            root.remove("include");
        }
        if removed_marker {
            // `top_logprobs` is only meaningful together with the removed include
            // marker. Leaving the number by itself still trips Muse validation.
            removed_marker |= root.remove("top_logprobs").is_some();
        }
    }
    removed_marker
}

fn normalize_replayed_function_call_arguments(body: &mut Value) -> bool {
    let Some(input) = body.get_mut("input").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut changed = false;
    for item in input {
        if json_type(item) != Some("function_call") {
            continue;
        }
        let Some(object) = item.as_object_mut() else {
            continue;
        };
        let valid = object
            .get("arguments")
            .and_then(Value::as_str)
            .is_some_and(|arguments| {
                let arguments = arguments.trim();
                !arguments.is_empty() && serde_json::from_str::<Value>(arguments).is_ok()
            });
        if !valid {
            object.insert("arguments".to_string(), Value::String("{}".to_string()));
            changed = true;
        }
    }
    changed
}

fn is_compaction_summary_turn(body: &Value) -> bool {
    body.get("input")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|item| json_type(item) == Some("message"))
        .flat_map(|item| {
            item.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .any(|text| text == super::codex_compaction::COMPACT_PROMPT)
}

fn is_additional_tools_item(item: &Value) -> bool {
    json_type(item) == Some("additional_tools")
}

fn collect_declared_tools(value: Option<&Value>, tools: &mut Vec<Value>) {
    if let Some(declared) = value.and_then(Value::as_array) {
        tools.extend(declared.iter().cloned());
    }
}

/// Promote `additional_tools` carriers into top-level tools and remove the
/// private carriers themselves. `tool_search_output` is intentionally left
/// untouched because OpenCode now accepts it.
fn promote_additional_tools(body: &mut Value) -> bool {
    let mut discovered = Vec::new();
    collect_declared_tools(body.get("additional_tools"), &mut discovered);
    if let Some(items) = body.get("input").and_then(Value::as_array) {
        for item in items.iter().filter(|item| is_additional_tools_item(item)) {
            collect_declared_tools(item.get("tools"), &mut discovered);
        }
    }

    let mut changed = false;
    if !discovered.is_empty() {
        if !body.get("tools").is_some_and(Value::is_array) {
            body["tools"] = json!([]);
            changed = true;
        }
        if let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) {
            for tool in discovered {
                changed |= merge_tool_declaration(tools, tool);
            }
        }
    }
    changed |= remove_additional_tool_carriers(body, true);
    changed
}

fn remove_additional_tool_carriers(body: &mut Value, remove_top_level: bool) -> bool {
    let mut changed = false;
    if remove_top_level {
        changed |= body
            .as_object_mut()
            .is_some_and(|root| root.remove("additional_tools").is_some());
    }
    if let Some(input) = body.get_mut("input").and_then(Value::as_array_mut) {
        let before = input.len();
        input.retain(|item| !is_additional_tools_item(item));
        changed |= input.len() != before;
    }
    changed
}

fn tool_identity(tool: &Value) -> (&str, &str) {
    (
        tool.get("type").and_then(Value::as_str).unwrap_or(""),
        tool.get("name").and_then(Value::as_str).unwrap_or(""),
    )
}

fn tool_dedup_key(tool: &Value) -> String {
    let (tool_type, name) = tool_identity(tool);
    if !tool_type.is_empty() && !name.is_empty() {
        return format!("{tool_type}\0{name}");
    }
    format!("json:{tool}")
}

fn namespace_children(tool: &Value) -> Vec<Value> {
    tool.get("tools")
        .or_else(|| tool.get("children"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn merge_tool_declaration(tools: &mut Vec<Value>, incoming: Value) -> bool {
    let key = tool_dedup_key(&incoming);
    let Some(existing) = tools
        .iter_mut()
        .find(|existing| tool_dedup_key(existing) == key)
    else {
        tools.push(incoming);
        return true;
    };
    if json_type(existing) == Some("namespace") && json_type(&incoming) == Some("namespace") {
        return merge_namespace_declaration(existing, &incoming);
    }
    false
}

fn merge_namespace_declaration(existing: &mut Value, incoming: &Value) -> bool {
    let mut children = namespace_children(existing);
    let mut seen: HashSet<String> = children.iter().map(tool_dedup_key).collect();
    let mut changed = false;
    for child in namespace_children(incoming) {
        if seen.insert(tool_dedup_key(&child)) {
            children.push(child);
            changed = true;
        }
    }
    if !changed {
        return false;
    }
    let Some(object) = existing.as_object_mut() else {
        return false;
    };
    object.remove("children");
    object.insert("tools".to_string(), Value::Array(children));
    true
}

fn empty_function_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {},
        "additionalProperties": false
    })
}

fn normalize_parameters_object(parameters: &mut Value) -> bool {
    let Some(object) = parameters.as_object_mut() else {
        let replacement = empty_function_parameters();
        if *parameters == replacement {
            return false;
        }
        *parameters = replacement;
        return true;
    };

    if object
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|schema_type| schema_type != "object")
    {
        let replacement = empty_function_parameters();
        if *parameters == replacement {
            return false;
        }
        *parameters = replacement;
        return true;
    }

    let mut changed = false;
    if object.get("type").and_then(Value::as_str) != Some("object") {
        object.insert("type".to_string(), Value::String("object".to_string()));
        changed = true;
    }
    if !object.get("properties").is_some_and(Value::is_object) {
        object.insert("properties".to_string(), json!({}));
        changed = true;
    }
    changed
}

fn normalize_function_tools(tools: &mut [Value]) -> bool {
    let mut changed = false;
    for tool in tools {
        match json_type(tool) {
            Some("function") => {
                if let Some(object) = tool.as_object_mut() {
                    let parameters = object
                        .entry("parameters".to_string())
                        .or_insert_with(empty_function_parameters);
                    changed |= normalize_parameters_object(parameters);
                }
            }
            Some("namespace") => {
                for key in ["tools", "children"] {
                    if let Some(children) = tool.get_mut(key).and_then(Value::as_array_mut) {
                        changed |= normalize_function_tools(children);
                    }
                }
            }
            _ => {}
        }
    }
    changed
}

fn normalize_function_tool_schemas(body: &mut Value) -> bool {
    body.get_mut("tools")
        .and_then(Value::as_array_mut)
        .is_some_and(|tools| normalize_function_tools(tools))
}

fn normalize_tool_search_schemas(body: &mut Value) -> bool {
    let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut changed = false;
    for tool in tools {
        if json_type(tool) != Some("tool_search") {
            continue;
        }
        let Some(object) = tool.as_object_mut() else {
            continue;
        };
        let parameters = object
            .entry("parameters".to_string())
            .or_insert_with(empty_function_parameters);
        changed |= normalize_parameters_object(parameters);
        let Some(parameters) = parameters.as_object_mut() else {
            continue;
        };

        let property_names = parameters
            .get("properties")
            .and_then(Value::as_object)
            .map(|properties| properties.keys().cloned().collect::<HashSet<_>>())
            .unwrap_or_default();
        if let Some(required) = parameters.get_mut("required").and_then(Value::as_array_mut) {
            let before = required.len();
            required.retain(|name| {
                name.as_str()
                    .is_some_and(|name| property_names.contains(name))
            });
            changed |= required.len() != before;
        } else {
            parameters.insert("required".to_string(), json!([]));
            changed = true;
        }
        if parameters.get("additionalProperties") != Some(&Value::Bool(false)) {
            parameters.insert("additionalProperties".to_string(), Value::Bool(false));
            changed = true;
        }
    }
    changed
}

fn normalize_tool_choice(body: &mut Value) -> bool {
    let Some(root) = body.as_object_mut() else {
        return false;
    };
    let has_tools = root
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    if !has_tools {
        let mut changed = false;
        changed |= root.remove("tool_choice").is_some();
        changed |= root.remove("parallel_tool_calls").is_some();
        return changed;
    }
    let Some(choice) = root.get("tool_choice") else {
        return false;
    };
    if choice.as_str() == Some("auto") {
        return false;
    }
    root.insert("tool_choice".to_string(), Value::String("auto".to_string()));
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixes_only_the_open_remaining_muse_request_gaps() {
        let mut body = json!({
            "model": "ccs-opencode-4/muse-spark-1.3-contributor",
            "access_programs": ["codex"],
            "external_web_access": true,
            "reasoning": { "effort": "minimal", "summary": "none" },
            "include": [
                "reasoning.encrypted_content",
                "message.output_text.logprobs"
            ],
            "top_logprobs": 1,
            "additional_tools": [{
                "type": "function",
                "name": "loaded"
            }],
            "tools": [{
                "type": "function",
                "name": "no_parameters"
            }, {
                "type": "tool_search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "limit": { "type": "number" },
                        "filters": { "type": "object" }
                    },
                    "required": ["query"]
                }
            }],
            "tool_choice": { "type": "function", "name": "no_parameters" },
            "input": [{
                "type": "additional_tools",
                "tools": [{
                    "type": "function",
                    "name": "from_input",
                    "parameters": {}
                }]
            }, {
                "type": "function_call",
                "call_id": "call_1",
                "name": "no_parameters",
                "arguments": ""
            }, {
                "type": "function_call_output",
                "output": "ok"
            }]
        });

        assert!(apply_opencode_muse_request_compat(&mut body));
        assert!(body.get("access_programs").is_none());
        assert!(body.get("external_web_access").is_none());
        assert_eq!(body["reasoning"], json!({ "effort": "minimal" }));
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert!(body.get("top_logprobs").is_none());
        assert_eq!(body["tool_choice"], "auto");
        assert!(body["tools"].as_array().unwrap().iter().all(|tool| {
            tool.get("name").and_then(Value::as_str) != Some("from_input")
                || tool.get("parameters").is_some()
        }));
        let names: Vec<&str> = body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str))
            .collect();
        assert!(names.contains(&"loaded"));
        assert!(names.contains(&"from_input"));
        assert!(body["input"]
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item.get("type").and_then(Value::as_str) != Some("additional_tools")));
        assert_eq!(body["input"][0]["arguments"], "{}");
        assert_eq!(body["input"][1]["type"], "function_call_output");
        assert_eq!(body["input"][1]["call_id"], "call_1");
        let loaded = body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == "loaded")
            .unwrap();
        assert_eq!(
            loaded["parameters"],
            json!({ "type": "object", "properties": {}, "additionalProperties": false })
        );
    }

    #[test]
    fn preserves_upstream_accepted_shapes() {
        let mut body = json!({
            "tools": [{
                "type": "namespace",
                "name": "mcp__codex_app",
                "tools": [{
                    "type": "function",
                    "name": "read_thread",
                    "parameters": { "type": "object", "properties": {} }
                }]
            }, {
                "type": "tool_search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "limit": { "type": "number" },
                        "filters": { "type": "object" }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            }],
            "input": [{
                "type": "tool_search_output",
                "tools": [{
                    "type": "function",
                    "name": "already_accepted",
                    "parameters": { "type": "object", "properties": {} }
                }]
            }],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": { "type": "object" }
                }
            },
            "tool_choice": "auto"
        });
        let before = body.clone();

        assert!(!apply_opencode_muse_request_compat(&mut body));
        assert_eq!(body, before);
        assert!(body["tools"][0].get("tools").is_some());
    }

    #[test]
    fn pairs_only_unambiguous_receipts_and_preserves_other_output_as_message() {
        let mut body = json!({
            "input": [{
                "type": "function_call",
                "call_id": "call_only",
                "name": "echo",
                "arguments": "{}"
            }, {
                "type": "function_call_output",
                "output": "matched"
            }, {
                "type": "custom_tool_call_output",
                "output": "cross-thread"
            }]
        });

        assert!(apply_opencode_muse_request_compat(&mut body));
        assert_eq!(body["input"][1]["type"], "function_call_output");
        assert_eq!(body["input"][1]["call_id"], "call_only");
        assert_eq!(body["input"][2]["type"], "message");
        assert!(body["input"][2]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("cross-thread"));
    }

    #[test]
    fn compaction_turn_drops_unsupported_tool_choice_shape() {
        let mut body = json!({
            "additional_tools": [{
                "type": "function",
                "name": "top_level_loaded",
                "parameters": { "type": "object", "properties": {} }
            }],
            "tools": [{
                "type": "function",
                "name": "shell",
                "parameters": { "type": "object", "properties": {} }
            }],
            "tool_choice": "none",
            "parallel_tool_calls": true,
            "input": [{
                "type": "additional_tools",
                "tools": [{
                    "type": "function",
                    "name": "input_loaded",
                    "parameters": { "type": "object", "properties": {} }
                }]
            }, {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": super::super::codex_compaction::COMPACT_PROMPT }]
            }]
        });

        assert!(apply_opencode_muse_request_compat(&mut body));
        assert!(body.get("additional_tools").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
        assert_eq!(body["input"].as_array().unwrap().len(), 1);
        assert_eq!(body["input"][0]["type"], "message");
    }

    #[test]
    fn removes_tool_choice_controls_when_no_tools_remain() {
        for tools in [None, Some(json!([]))] {
            let mut body = json!({
                "tool_choice": "auto",
                "parallel_tool_calls": true,
                "input": []
            });
            if let Some(tools) = tools {
                body["tools"] = tools;
            }

            assert!(apply_opencode_muse_request_compat(&mut body));
            assert!(body.get("tool_choice").is_none());
            assert!(body.get("parallel_tool_calls").is_none());
        }
    }

    #[test]
    fn leaves_unaffected_request_byte_identical() {
        let mut body = json!({
            "model": "muse-spark-1.3-contributor",
            "input": [{ "type": "message", "role": "user", "content": "hello" }]
        });
        let before = body.clone();

        assert!(!apply_opencode_muse_request_compat(&mut body));
        assert_eq!(body, before);
    }
}
