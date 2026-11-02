//! Native Responses compatibility for Codex's client-side `tool_search` tool.
//!
//! OpenCode/Muse accepts native Responses and `tool_search`, but its validator
//! is stricter than OpenAI's in several places. Keep those rewrites isolated
//! here so the native passthrough stays byte-identical for every other model.

use std::collections::HashMap;

use serde_json::Value;

use super::transform_codex_chat::flatten_namespace_tool_name;
use super::transform_codex_responses_namespace::NamespacedName;
use crate::proxy::error::ProxyError;

const MUSE_UNSUPPORTED_TOP_LEVEL_FIELDS: &[&str] = &["access_programs", "external_web_access"];
const MUSE_UNSUPPORTED_INCLUDE_MARKERS: &[&str] =
    &["message.output_text.logprobs", "output_text.logprobs"];

/// Apply every request-side rewrite OpenCode/Muse needs. The caller gates this
/// by provider host, native `wire_api`, and the final outbound model.
pub(crate) fn apply_opencode_muse_request_compat(body: &mut Value) -> Result<bool, ProxyError> {
    let mut changed = false;
    if let Some(root) = body.as_object_mut() {
        for field in MUSE_UNSUPPORTED_TOP_LEVEL_FIELDS {
            changed |= root.remove(*field).is_some();
        }
    }
    changed |= strip_unsupported_include_markers(body);

    if is_compaction_summary_turn(body) {
        let restore_map = super::transform_codex_responses_namespace::namespace_restore_map(body);
        changed |= normalize_opencode_muse_tool_calls(body, &restore_map);
        changed |= normalize_tool_search_output_carriers(body);
        changed |= remove_additional_tool_input_carriers(body);
        if let Some(root) = body.as_object_mut() {
            changed |= root.remove("tools").is_some();
        }
        changed |= normalize_tool_choice_for_muse(body);
        return Ok(changed);
    }

    changed |= promote_tool_search_output_tools(body)?;

    let restore_map = super::transform_codex_responses_namespace::namespace_restore_map(body);
    changed |= super::transform_codex_responses_namespace::flatten_request_namespaces(body)?;
    changed |= normalize_opencode_muse_tool_calls(body, &restore_map);
    changed |= normalize_tool_search_output_carriers(body);
    changed |= remove_additional_tool_input_carriers(body);
    changed |= normalize_function_tool_schemas(body);
    changed |= normalize_tool_search_schemas(body);
    changed |= normalize_web_search_tools(body);

    if has_structured_output(body) {
        changed |= remove_tool_search_declarations(body);
    }
    changed |= normalize_tool_choice_for_muse(body);
    Ok(changed)
}

fn is_compaction_summary_turn(body: &Value) -> bool {
    body.get("input")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .flat_map(|item| {
            item.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .any(|text| text == super::codex_compaction::COMPACT_PROMPT)
}

/// Promote tools returned by the client-side `tool_search` into the request's
/// top-level `tools` array. The OpenAI backend understands dynamically loaded
/// namespace tools in `tool_search_output`; OpenCode/Muse does not, and will
/// otherwise emit a namespace-qualified call without its parameter schema.
pub(crate) fn promote_tool_search_output_tools(body: &mut Value) -> Result<bool, ProxyError> {
    let mut discovered = Vec::new();
    collect_input_declared_tools(body.get("input"), &mut discovered);
    collect_declared_tools_array(body.get("additional_tools"), &mut discovered);
    if discovered.is_empty() {
        return Ok(false);
    }

    let tools = body.as_object_mut().and_then(|root| {
        root.entry("tools")
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
    });
    let Some(tools) = tools else {
        return Ok(false);
    };

    let mut changed = false;
    for tool in discovered {
        changed |= merge_tool_declaration(tools, tool)?;
    }
    Ok(changed)
}

/// Repair namespace-qualified or malformed `function_call` history produced by
/// Muse after a deferred tool was discovered. OpenCode validates replayed call
/// arguments before the Codex client can execute them, so an empty/invalid
/// string or `namespace.child` name would otherwise pin the thread to HTTP 400.
pub(crate) fn normalize_opencode_muse_tool_calls(
    body: &mut Value,
    restore_map: &HashMap<String, NamespacedName>,
) -> bool {
    let Some(items) = body.get_mut("input").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut changed = false;
    for item in items {
        let Some(object) = item.as_object_mut() else {
            continue;
        };
        if object.get("type").and_then(Value::as_str) == Some("function_call") {
            changed |= rewrite_tool_name(object, restore_map);
            changed |= rewrite_tool_arguments(object);
        }
    }
    changed
}

fn rewrite_tool_name(
    object: &mut serde_json::Map<String, Value>,
    restore_map: &HashMap<String, NamespacedName>,
) -> bool {
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return false;
    };
    let Some((namespace, child)) = name.split_once('.') else {
        return false;
    };
    let flat = flatten_namespace_tool_name(namespace, child);
    if !restore_map.contains_key(&flat) {
        return false;
    }
    object.insert("name".to_string(), Value::String(flat));
    true
}

fn rewrite_tool_arguments(object: &mut serde_json::Map<String, Value>) -> bool {
    let Some(arguments) = object.get_mut("arguments") else {
        object.insert("arguments".to_string(), Value::String("{}".to_string()));
        return true;
    };
    match arguments {
        Value::String(text)
            if !text.trim().is_empty() && serde_json::from_str::<Value>(text).is_ok() =>
        {
            false
        }
        _ => {
            *arguments = Value::String("{}".to_string());
            true
        }
    }
}

fn collect_input_declared_tools(input: Option<&Value>, tools: &mut Vec<Value>) {
    let Some(items) = input.and_then(Value::as_array) else {
        return;
    };
    for item in items {
        if matches!(
            item.get("type").and_then(Value::as_str),
            Some("tool_search_output" | "additional_tools")
        ) {
            collect_declared_tools_array(item.get("tools"), tools);
        }
    }
}

fn collect_declared_tools_array(value: Option<&Value>, tools: &mut Vec<Value>) {
    if let Some(declared) = value.and_then(Value::as_array) {
        tools.extend(declared.iter().cloned());
    }
}

fn normalize_tool_search_output_carriers(body: &mut Value) -> bool {
    let Some(items) = body.get_mut("input").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut changed = false;
    for item in items {
        if item.get("type").and_then(Value::as_str) != Some("tool_search_output") {
            continue;
        }
        let Some(tools) = item.get("tools").and_then(Value::as_array) else {
            continue;
        };
        let mut flattened = Vec::new();
        for tool in tools {
            if tool.get("type").and_then(Value::as_str) == Some("namespace") {
                let Some(namespace) = tool.get("name").and_then(Value::as_str) else {
                    continue;
                };
                for child in namespace_children(tool) {
                    if child.get("type").and_then(Value::as_str) != Some("function") {
                        continue;
                    }
                    if let Some(name) = child
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                    {
                        let mut child = child;
                        if let Some(object) = child.as_object_mut() {
                            object.insert(
                                "name".to_string(),
                                Value::String(flatten_namespace_tool_name(namespace, &name)),
                            );
                        }
                        flattened.push(child);
                    }
                }
            } else if tool.get("type").and_then(Value::as_str) == Some("function") {
                flattened.push(tool.clone());
            }
        }
        if &flattened != tools {
            if let Some(object) = item.as_object_mut() {
                object.insert("tools".to_string(), Value::Array(flattened));
                changed = true;
            }
        }
    }
    changed
}

fn merge_tool_declaration(tools: &mut Vec<Value>, tool: Value) -> Result<bool, ProxyError> {
    let identity = tool_identity(&tool);
    if let Some(existing) = tools
        .iter_mut()
        .find(|existing| tool_identity(existing) == identity)
    {
        if tool.get("type").and_then(Value::as_str) == Some("namespace") {
            return Ok(merge_namespace_declaration(existing, &tool));
        }
        return Ok(false);
    }
    tools.push(tool);
    Ok(true)
}

fn tool_identity(tool: &Value) -> (&str, &str) {
    (
        tool.get("type").and_then(Value::as_str).unwrap_or(""),
        tool.get("name").and_then(Value::as_str).unwrap_or(""),
    )
}

fn merge_namespace_declaration(existing: &mut Value, incoming: &Value) -> bool {
    let mut children = namespace_children(existing);
    let mut changed = false;
    for child in namespace_children(incoming) {
        if !children
            .iter()
            .any(|existing| tool_identity(existing) == tool_identity(&child))
        {
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

fn namespace_children(tool: &Value) -> Vec<Value> {
    tool.get("tools")
        .or_else(|| tool.get("children"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn strip_unsupported_include_markers(body: &mut Value) -> bool {
    let Some(include) = body.get_mut("include").and_then(Value::as_array_mut) else {
        return false;
    };
    let before = include.len();
    include.retain(|marker| {
        !marker
            .as_str()
            .is_some_and(|marker| MUSE_UNSUPPORTED_INCLUDE_MARKERS.contains(&marker))
    });
    let changed = include.len() != before;
    let empty = include.is_empty();
    if empty {
        if let Some(body) = body.as_object_mut() {
            body.remove("include");
        }
    }
    changed
}

fn remove_additional_tool_input_carriers(body: &mut Value) -> bool {
    let mut changed = false;
    if let Some(root) = body.as_object_mut() {
        changed |= root.remove("additional_tools").is_some();
    }
    if let Some(input) = body.get_mut("input").and_then(Value::as_array_mut) {
        let before = input.len();
        input.retain(|item| item.get("type").and_then(Value::as_str) != Some("additional_tools"));
        changed |= input.len() != before;
    }
    changed
}

fn empty_function_parameters() -> Value {
    serde_json::json!({
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
        object.insert("properties".to_string(), serde_json::json!({}));
        changed = true;
    }
    changed
}

fn normalize_function_tool_schemas(body: &mut Value) -> bool {
    let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut changed = false;
    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            continue;
        }
        let Some(object) = tool.as_object_mut() else {
            continue;
        };
        let parameters = object
            .entry("parameters".to_string())
            .or_insert_with(empty_function_parameters);
        changed |= normalize_parameters_object(parameters);
    }
    changed
}

fn normalize_tool_search_schemas(body: &mut Value) -> bool {
    let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut changed = false;
    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("tool_search") {
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
        let required: std::collections::HashSet<String> = parameters
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        if let Some(properties) = parameters
            .get_mut("properties")
            .and_then(Value::as_object_mut)
        {
            let before = properties.len();
            properties.retain(|name, _| name != "limit" && required.contains(name));
            changed |= properties.len() != before;
        }
        let property_names = parameters
            .get("properties")
            .and_then(Value::as_object)
            .map(|properties| {
                properties
                    .keys()
                    .cloned()
                    .collect::<std::collections::HashSet<_>>()
            })
            .unwrap_or_default();
        if let Some(required_values) = parameters.get_mut("required").and_then(Value::as_array_mut)
        {
            let before = required_values.len();
            required_values.retain(|value| {
                value
                    .as_str()
                    .is_some_and(|name| name != "limit" && property_names.contains(name))
            });
            changed |= required_values.len() != before;
        } else {
            parameters.insert("required".to_string(), Value::Array(Vec::new()));
            changed = true;
        }
        if parameters.get("additionalProperties") != Some(&Value::Bool(false)) {
            parameters.insert("additionalProperties".to_string(), Value::Bool(false));
            changed = true;
        }
    }
    changed
}

fn normalize_web_search_tools(body: &mut Value) -> bool {
    let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut changed = false;
    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("web_search") {
            continue;
        }
        if let Some(object) = tool.as_object_mut() {
            changed |= object.remove("search_content_types").is_some();
        }
    }
    changed
}

fn has_structured_output(body: &Value) -> bool {
    body.pointer("/text/format/type").and_then(Value::as_str) == Some("json_schema")
}

fn remove_tool_search_declarations(body: &mut Value) -> bool {
    let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) else {
        return false;
    };
    let before = tools.len();
    tools.retain(|tool| tool.get("type").and_then(Value::as_str) != Some("tool_search"));
    tools.len() != before
}

fn normalize_tool_choice_for_muse(body: &mut Value) -> bool {
    let has_tools = body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    let Some(object) = body.as_object_mut() else {
        return false;
    };
    if !has_tools {
        let mut changed = object.remove("tool_choice").is_some();
        changed |= object.remove("parallel_tool_calls").is_some();
        return changed;
    }

    let needs_auto = object
        .get("tool_choice")
        .is_some_and(|choice| choice.as_str() != Some("auto"));
    if needs_auto {
        object.insert("tool_choice".to_string(), Value::String("auto".to_string()));
    }
    needs_auto
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn promotes_and_merges_incremental_namespace_children() {
        let mut body = json!({
            "tools": [
                { "type": "tool_search", "parameters": { "type": "object", "properties": {} } }
            ],
            "input": [{
                "type": "tool_search_output",
                "call_id": "search-1",
                "tools": [{
                    "type": "namespace",
                    "name": "mcp__codex_app",
                    "tools": [{
                        "type": "function",
                        "name": "read_thread",
                        "parameters": { "type": "object", "properties": {} }
                    }]
                }]
            }, {
                "type": "tool_search_output",
                "call_id": "search-2",
                "tools": [{
                    "type": "namespace",
                    "name": "mcp__codex_app",
                    "tools": [{
                        "type": "function",
                        "name": "list_threads",
                        "parameters": { "type": "object", "properties": {} }
                    }]
                }]
            }]
        });

        assert!(promote_tool_search_output_tools(&mut body).unwrap());
        assert_eq!(body["tools"].as_array().unwrap().len(), 2);
        assert_eq!(body["tools"][1]["tools"].as_array().unwrap().len(), 2);
        assert!(!promote_tool_search_output_tools(&mut body).unwrap());

        assert!(
            super::super::transform_codex_responses_namespace::flatten_request_namespaces(
                &mut body
            )
            .unwrap()
        );
        assert!(body["tools"].as_array().unwrap().iter().any(|tool| {
            tool.get("type").and_then(Value::as_str) == Some("function")
                && tool.get("name").and_then(Value::as_str) == Some("mcp__codex_app__read_thread")
        }));
        assert!(body["tools"].as_array().unwrap().iter().any(|tool| {
            tool.get("type").and_then(Value::as_str) == Some("function")
                && tool.get("name").and_then(Value::as_str) == Some("mcp__codex_app__list_threads")
        }));
    }

    #[test]
    fn repairs_namespace_qualified_call_and_empty_arguments() {
        let body = json!({
            "input": [{
                "type": "tool_search_output",
                "tools": [{
                    "type": "namespace",
                    "name": "mcp__codex_app",
                    "tools": [{
                        "type": "function",
                        "name": "read_thread",
                        "parameters": {}
                    }]
                }]
            }, {
                "type": "function_call",
                "name": "mcp__codex_app.read_thread",
                "arguments": ""
            }]
        });
        let restore_map =
            super::super::transform_codex_responses_namespace::namespace_restore_map(&body);
        let mut body = body;

        assert!(normalize_opencode_muse_tool_calls(&mut body, &restore_map));
        assert_eq!(body["input"][1]["name"], "mcp__codex_app__read_thread");
        assert_eq!(body["input"][1]["arguments"], "{}");
    }

    #[test]
    fn does_not_rewrite_function_call_shaped_business_output() {
        let mut body = json!({
            "input": [{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {
                    "type": "function_call",
                    "name": "mcp__codex_app.read_thread",
                    "arguments": ""
                }
            }]
        });
        let restore_map =
            super::super::transform_codex_responses_namespace::namespace_restore_map(&body);

        assert!(!normalize_opencode_muse_tool_calls(&mut body, &restore_map));
        assert_eq!(
            body["input"][0]["output"]["name"],
            "mcp__codex_app.read_thread"
        );
        assert_eq!(body["input"][0]["output"]["arguments"], "");
    }

    #[test]
    fn full_muse_tool_search_round_trip_flattens_and_restores_namespace() {
        let mut request = json!({
            "model": "muse-spark-1.3-contributor",
            "tools": [{
                "type": "tool_search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "limit": { "type": "number" }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }
            }],
            "input": [{
                "type": "tool_search_call",
                "call_id": "search-1",
                "arguments": { "query": "read thread" }
            }, {
                "type": "tool_search_output",
                "call_id": "search-1",
                "tools": [{
                    "type": "namespace",
                    "name": "mcp__codex_app",
                    "tools": [{
                        "type": "function",
                        "name": "read_thread",
                        "parameters": {
                            "type": "object",
                            "properties": { "threadId": { "type": "string" } },
                            "required": ["threadId"],
                            "additionalProperties": false
                        }
                    }]
                }]
            }]
        });
        let restore_map =
            super::super::transform_codex_responses_namespace::namespace_restore_map(&request);

        assert!(apply_opencode_muse_request_compat(&mut request).unwrap());
        let read_thread = request["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == "mcp__codex_app__read_thread")
            .unwrap();
        assert_eq!(read_thread["parameters"]["required"], json!(["threadId"]));

        let mut response_event = json!({
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "name": "mcp__codex_app__read_thread",
                "arguments": "{\"threadId\":\"thread-1\"}"
            }
        });
        assert!(
            super::super::transform_codex_responses_namespace::restore_sse_event_namespaces(
                &mut response_event,
                &restore_map
            )
        );
        assert_eq!(response_event["item"]["namespace"], "mcp__codex_app");
        assert_eq!(response_event["item"]["name"], "read_thread");
        assert_eq!(
            response_event["item"]["arguments"],
            "{\"threadId\":\"thread-1\"}"
        );
    }

    #[test]
    fn removes_optional_tool_search_properties_and_preserves_function_fields() {
        let mut body = json!({
            "tools": [
                {
                    "type": "tool_search",
                    "execution": "client",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" },
                            "limit": { "type": "number" }
                        },
                        "required": ["query"],
                        "additionalProperties": false
                    }
                },
                {
                    "type": "function",
                    "name": "keep_me",
                    "parameters": {
                        "type": "object",
                        "properties": { "limit": { "type": "number" } }
                    }
                }
            ]
        });

        assert!(apply_opencode_muse_request_compat(&mut body).unwrap());
        assert!(body["tools"][0]["parameters"]["properties"]
            .get("limit")
            .is_none());
        assert_eq!(
            body["tools"][0]["parameters"]["properties"]["query"]["type"],
            "string"
        );
        assert!(body["tools"][1]["parameters"]["properties"]
            .get("limit")
            .is_some());
        assert!(!apply_opencode_muse_request_compat(&mut body).unwrap());
    }

    #[test]
    fn removes_additional_tools_carriers_and_fills_missing_function_parameters() {
        let mut body = json!({
            "access_programs": ["codex"],
            "external_web_access": true,
            "include": [
                "reasoning.encrypted_content",
                "message.output_text.logprobs"
            ],
            "tools": [{
                "type": "function",
                "name": "no_parameters"
            }],
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{
                        "type": "function",
                        "name": "loaded",
                        "parameters": {}
                    }]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "hi" }]
                }
            ]
        });

        assert!(apply_opencode_muse_request_compat(&mut body).unwrap());
        assert!(body.get("access_programs").is_none());
        assert!(body.get("external_web_access").is_none());
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(body["input"].as_array().unwrap().len(), 1);
        assert_eq!(body["input"][0]["type"], "message");
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["parameters"]["type"], "object");
        assert_eq!(tools[1]["name"], "loaded");
    }

    #[test]
    fn structured_output_drops_only_tool_search_declaration() {
        let mut body = json!({
            "tools": [
                {
                    "type": "tool_search",
                    "description": "Search tools",
                    "parameters": {
                        "type": "object",
                        "properties": { "query": { "type": "string" } },
                        "required": ["query"],
                        "additionalProperties": false
                    }
                },
                {
                    "type": "function",
                    "name": "shell",
                    "parameters": { "type": "object", "properties": {} }
                }
            ],
            "tool_choice": { "type": "required" },
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "result",
                    "schema": { "type": "object" }
                }
            }
        });

        assert!(apply_opencode_muse_request_compat(&mut body).unwrap());
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "shell");
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn normalizes_non_auto_choice_and_removes_choice_without_tools() {
        for choice in [
            json!("none"),
            json!("required"),
            json!({ "type": "function", "name": "shell" }),
            json!({ "type": "auto" }),
        ] {
            let mut body = json!({
                "tools": [{ "type": "function", "name": "shell" }],
                "tool_choice": choice
            });
            assert!(apply_opencode_muse_request_compat(&mut body).unwrap());
            assert_eq!(body["tool_choice"], "auto");
        }

        let mut no_tools = json!({
            "tools": [],
            "tool_choice": "none",
            "parallel_tool_calls": true
        });
        assert!(apply_opencode_muse_request_compat(&mut no_tools).unwrap());
        assert!(no_tools.get("tool_choice").is_none());
        assert!(no_tools.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn compaction_summary_turn_never_sends_the_unsupported_none_choice() {
        let mut body = json!({
            "model": "muse-spark-1.3-contributor",
            "tools": [{
                "type": "function",
                "name": "shell",
                "parameters": { "type": "object", "properties": {} }
            }],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "hi" }]
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "shell",
                    "arguments": "{\"command\":\"echo hi\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "hi"
                },
                { "type": "compaction_trigger" }
            ]
        });

        assert!(super::super::codex_compaction::prepare_native_third_party_request(&mut body));
        assert_eq!(body["tool_choice"], "none");
        assert!(apply_opencode_muse_request_compat(&mut body).unwrap());
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
        assert!(body["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item.get("type").and_then(Value::as_str) == Some("function_call") }));
    }

    #[test]
    fn only_scans_direct_input_tool_carriers() {
        let mut body = json!({
            "tools": [{ "type": "tool_search", "parameters": {} }],
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "hi"
                }],
                "metadata": {
                    "results": [{
                        "type": "additional_tools",
                        "tools": [{
                            "type": "function",
                            "name": "smuggled"
                        }]
                    }]
                }
            }]
        });

        assert!(!promote_tool_search_output_tools(&mut body).unwrap());
        assert!(!body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| { tool.get("name").and_then(Value::as_str) == Some("smuggled") }));
    }

    #[test]
    fn flattens_discovered_namespaces_in_the_output_carrier_too() {
        let mut body = json!({
            "tools": [{
                "type": "tool_search",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": { "query": { "type": "string" } },
                    "required": ["query"],
                    "additionalProperties": false
                }
            }],
            "input": [{
                "type": "tool_search_call",
                "call_id": "search-1",
                "arguments": { "query": "echo" }
            }, {
                "type": "tool_search_output",
                "call_id": "search-1",
                "tools": [{
                    "type": "namespace",
                    "name": "mcp__probe",
                    "tools": [{
                        "type": "function",
                        "name": "echo",
                        "parameters": {
                            "type": "object",
                            "properties": { "text": { "type": "string" } },
                            "required": ["text"],
                            "additionalProperties": false
                        }
                    }]
                }]
            }]
        });

        assert!(apply_opencode_muse_request_compat(&mut body).unwrap());
        let output_tools = body["input"][1]["tools"].as_array().unwrap();
        assert_eq!(output_tools.len(), 1);
        assert_eq!(output_tools[0]["type"], "function");
        assert_eq!(output_tools[0]["name"], "mcp__probe__echo");
        assert!(output_tools[0].get("namespace").is_none());
    }
}
