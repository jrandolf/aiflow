use serde_json::Value;

/// Parses a possibly incomplete JSON string, attempting to repair and deserialize it into a `serde_json::Value`.
///
/// # Arguments
///
/// * `input` - The JSON string to parse (may be incomplete or malformed).
///
/// # Returns
///
/// * `Ok(Value)` if parsing and repair succeed.
/// * `Err(anyhow::Error)` if the input cannot be repaired or parsed as JSON.
pub fn parse_incomplete_json(input: &str) -> Result<Value, anyhow::Error> {
    let value = repair_json::repair(input)?;
    Ok(serde_json::from_str(&value)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_json() {
        let input = r#"{"key": "value", "num": 42}"#;
        let result = parse_incomplete_json(input);
        let value = result.expect("valid JSON should parse");
        assert_eq!(
            value.get("key"),
            Some(&serde_json::Value::String(String::from("value")))
        );
        assert_eq!(value.get("num"), Some(&serde_json::Value::from(42_i64)));
    }

    #[test]
    fn repairs_and_parses_incomplete_json() {
        let input = r#"{"key": "value", "num": 42"#; // missing closing brace
        let result = parse_incomplete_json(input);
        let value = result.expect("repairable JSON should parse");
        assert_eq!(
            value.get("key"),
            Some(&serde_json::Value::String(String::from("value")))
        );
        assert_eq!(value.get("num"), Some(&serde_json::Value::from(42_i64)));
    }

    #[test]
    fn fails_on_unrepairable_json() {
        let input = r#"{\"key": [1, 2]"#; // Completely invalid JSON
        let result = parse_incomplete_json(input);
        assert!(result.is_err(), "Unrepairable JSON should return an error");
    }
}
